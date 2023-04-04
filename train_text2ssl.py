import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import librosa
import logging

import ssl2wav

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from data_utils import (
    TextSSLSpeakerLoader,
    TextSSLSpeakerCollate,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextSSLSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextSSLSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextSSLSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    net_g = DDP(net_g, device_ids=[rank])
    ssl2wav_model = ssl2wav.get_model(hps.data.ssl2wav_model_name).cuda(0)

    skip_optimizer = True
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, ssl2wav_model], [optim_g], [scheduler_g], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, None], [optim_g], [scheduler_g], scaler,
                               [train_loader, None], None, None)
        scheduler_g.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, ssl2wav_model = nets
    optim_g = optims[0]
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    for batch_idx, (x, x_lengths,lang, ssl_content, ssl_lengths, speakers) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        ssl_content, ssl_lengths = ssl_content.cuda(rank, non_blocking=True), ssl_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        lang = lang.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, lang, ssl_content, ssl_lengths, speakers)

            # Generator
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_ssl = F.l1_loss(y_hat, ssl_content) * hps.train.c_ssl
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_gen_all = loss_ssl + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_ssl, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr,
                               "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    { "loss/g/ssl": loss_ssl, "loss/g/dur": loss_dur,
                     "loss/g/kl": loss_kl})


                image_dict = {
                    "all/ssl_org": utils.plot_spectrogram_to_numpy(ssl_content[0].data.cpu().numpy()),
                    "all/ssl_gen": utils.plot_spectrogram_to_numpy(y_hat[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, ssl2wav_model,y_hat)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval, ssl2wav_model, train_rec):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        # y_train_rec, sr = ssl2wav.ssl2wav(hps.data.ssl2wav_model_name, ssl2wav_model, train_rec.detach()[:1, :, :],
        #                                   None)
        # audio_dict.update({
        #     f"gen/audio_tr_rec": y_train_rec
        # })
        for batch_idx,  (x, x_lengths,lang, ssl_content, ssl_lengths, speakers) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            ssl_content, ssl_lengths = ssl_content.cuda(0), ssl_lengths.cuda(0)
            speakers = speakers.cuda(0)
            lang = lang.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            ssl_content = ssl_content[:1]
            ssl_lengths = ssl_lengths[:1]
            speakers = speakers[:1]

            ssl_pred, attn, mask, *_ = generator.module.infer(x, x_lengths, lang, max_len=1000,sid=speakers)

            # image_dict.update({
            #     f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
            # })
            y_hat, sr = ssl2wav.ssl2wav(hps.data.ssl2wav_model_name, ssl2wav_model, ssl_pred, speakers)
            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat
            })
            y_rec, sr = ssl2wav.ssl2wav(hps.data.ssl2wav_model_name, ssl2wav_model, ssl_content, speakers)
            # image_dict.update({f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({f"rec/audio_{batch_idx}": y_rec})


    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=sr
    )
    generator.train()


if __name__ == "__main__":
    main()
