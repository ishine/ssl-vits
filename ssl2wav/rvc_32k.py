import torch

import utils
from models import SynthesizerTrnMs256NSFsid_nono
import torch.nn.functional as F

ssl_type = "contentvec"
hps = utils.get_hparams_from_file("./configs/32k.json")
pth = "pretrained/G_56000.pth"
tgt_sid = 11

def get_model():
    net_g = SynthesizerTrnMs256NSFsid_nono(hps.data.filter_length // 2 + 1,
                                           hps.train.segment_size // hps.data.hop_length,
                                           **hps.model,
                                           is_half=hps.train.fp16_run)
    _ = utils.load_checkpoint(pth, net_g, None)
    return net_g

def ssl2wav(model, ssl_content, sid):
    """
    Args:
        ssl_content (torch.Tensor): The SSL content tensor, with shape [1, 256, frame_len].
    Returns:
        audio (torch.Tensor): The audio waveform numpy array, with shape [wav_length].
    """
    feats = F.interpolate(ssl_content, scale_factor=2).permute(0, 2, 1)
    phone_lengths = torch.LongTensor([feats.size(1)]).to(feats.device)
    # sid = torch.LongTensor([tgt_sid]).to(feats.device)
    audio = model.infer(feats, phone_lengths, sid)[0][0, 0].data.cpu().float().numpy()
    return audio, hps.data.sampling_rate


