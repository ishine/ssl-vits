import math
import multiprocessing
import os
import argparse
from pathlib import Path
from random import shuffle

import soundfile
import torch
from glob import glob
from tqdm import tqdm

import ssl_extract
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np

hps = utils.get_hparams_from_file("configs/32k.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
ssl_base_path = f"dataset/{hps.data.ssl_type}"
preprocessed_wav_path = f"dataset/{hps.data.preprocessed_wav_path}"


def process_one(file_path, model):
    path = Path(file_path)
    name = path.stem

    spk_name = path.parent.name

    ssl_path = f"{ssl_base_path}/{spk_name}/{name}.pt"
    wav_path = f"{preprocessed_wav_path}/{spk_name}/{name}.wav"

    if not os.path.exists(ssl_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k, sr = librosa.load(path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = ssl_extract.get_ssl_content(hps.data.ssl_type, model, wav_16k_tensor=wav16k)
        torch.save(ssl_content.cpu(), ssl_path)
    if not os.path.exists(wav_path):
        wav, sr = librosa.load(path, sr=hps.data.sampling_rate)
        soundfile.write(wav_path, wav, sr)


def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl_model = ssl_extract.get_ssl_model(hps.data.ssl_type).to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/raw", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    os.makedirs(ssl_base_path, exist_ok=True)
    os.makedirs(preprocessed_wav_path, exist_ok=True)
    shuffle(filenames)
    filelist = []
    for file in filenames:
        path = Path(file)
        spk_name = path.parent.name
        name = path.stem
        ssl_base = f"{ssl_base_path}/{spk_name}"
        wav_base = f"{preprocessed_wav_path}/{spk_name}"
        os.makedirs(ssl_base, exist_ok=True)
        os.makedirs(wav_base, exist_ok=True)
        filelist.append([name, spk_name])
    with open(hps.data.training_files, "w") as f:
        for name, spk_name in filelist[:-2]:
            f.write(f"{name}|{spk_name}\n")
    with open(hps.data.validation_files, "w") as f:
        for name, spk_name in filelist[-2:]:
            f.write(f"{name}|{spk_name}\n")

    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()
