import soundfile
import torch.cuda

import ssl2wav
import ssl_extract
import utils
import logging
logging.getLogger("numba").setLevel(logging.INFO)
ssl_type = "contentvec"
src_path = "/Users/Shared/原音频2.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"
ssl2wav_model_name = "rvc_32k"
ssl2wav_model = ssl2wav.get_model(ssl2wav_model_name).to(device)

ssl_model = ssl_extract.get_ssl_model(ssl_type).to(device)
wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000).to(device)
ssl_content = ssl_extract.get_ssl_content(ssl_type, ssl_model, wav_16k_tensor)

audio, sr = ssl2wav.ssl2wav(ssl2wav_model_name, ssl2wav_model, ssl_content, None)
soundfile.write("out.wav", audio, sr)