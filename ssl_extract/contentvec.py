import torch
from fairseq import checkpoint_utils


def get_model():
  vec_path = "ssl_extract/hubert_base.pt"
  print("load model(s) from {}".format(vec_path))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [vec_path],
    suffix="",
  )
  model = models[0]
  model.eval()
  return model


def get_content(hmodel, wav_16k_tensor):
  feats = wav_16k_tensor
  if feats.dim() == 2:  # double channels
    feats = feats.mean(-1)
  assert feats.dim() == 1, feats.dim()
  feats = feats.view(1, -1)
  padding_mask = torch.BoolTensor(feats.shape).fill_(False)
  inputs = {
    "source": feats.to(wav_16k_tensor.device),
    "padding_mask": padding_mask.to(wav_16k_tensor.device),
    "output_layer": 9,  # layer 9
  }
  with torch.no_grad():
    logits = hmodel.extract_features(**inputs)
    feats = hmodel.final_proj(logits[0])
  return feats.transpose(1, 2)
