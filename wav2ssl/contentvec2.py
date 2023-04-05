import torch
from torch import nn
from transformers import HubertModel

import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

class HubertModelWithFinalProj(HubertModel):
  def __init__(self, config):
    super().__init__(config)

    # The final projection layer is only used for backward compatibility.
    # Following https://github.com/auspicious3000/contentvec/issues/6
    # Remove this layer is necessary to achieve the desired outcome.
    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def get_model():
  model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
  return model


def get_content(hmodel, wav_16k_tensor):
  with torch.no_grad():
      feats = hmodel(wav_16k_tensor.unsqueeze(0))["last_hidden_state"]
  return feats.transpose(1,2)

if __name__ == '__main__':
    model = get_model()
    src_path = "/Users/Shared/原音频2.wav"
    wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000)
    feats = get_content(model,wav_16k_tensor)
    print(feats.shape)