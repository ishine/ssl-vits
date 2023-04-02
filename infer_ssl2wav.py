import soundfile
import torch.cuda

import ssl_extract
import utils
from models import SynthesizerTrnMs256NSFsid_nono
import torch.nn.functional as F

src_path = "/Users/xingyijin/Downloads/输入源.wav"
ssl_type = "contentvec"
device = "cuda" if torch.cuda.is_available() else "cpu"
hps = utils.get_hparams_from_file("./configs/32k.json")
net_g = SynthesizerTrnMs256NSFsid_nono(hps.data.filter_length // 2 + 1,
                               hps.train.segment_size // hps.data.hop_length,
                               **hps.model,
                               is_half=hps.train.fp16_run).to(device)
_ = utils.load_checkpoint("pretrained/G32k.pth", net_g, None)

ssl_model = ssl_extract.get_ssl_model(ssl_type).to(device)
wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000).to(device)
ssl_content = ssl_extract.get_ssl_content(ssl_type, ssl_model, wav_16k_tensor)
feats = F.interpolate(ssl_content, scale_factor=2).permute(0, 2, 1)
phone_lengths = torch.LongTensor([feats.size(1)]).to(device)
sid = torch.LongTensor([100]).to(device)

audio = net_g.infer(feats, phone_lengths, sid)[0][0,0].data.cpu().float().numpy()
soundfile.write("out.wav", audio, hps.data.sampling_rate)