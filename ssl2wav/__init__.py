from . import rvc_32k

ssl2wav_model_map = {
    "rvc_32k": rvc_32k
}

def get_model(ssl2wav_model_name):
    return ssl2wav_model_map[ssl2wav_model_name].get_model()

def ssl2wav(ssl2wav_model_name, model, ssl_content, sid):
    """
    Args:
        ssl_content (torch.Tensor): The SSL content tensor, with shape [1, 256, frame_len].
    Returns:
        audio (torch.Tensor): The audio waveform numpy array, with shape [wav_length].
    """
    return ssl2wav_model_map[ssl2wav_model_name].ssl2wav(model, ssl_content, sid)


