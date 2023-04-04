from . import contentvec, contentvec2

ssl_map = {
    "contentvec": contentvec,
    "contentvec2": contentvec2
}

def get_ssl_model(ssl_type):
    assert ssl_type in ssl_map, "ssl type not supported"
    return ssl_map[ssl_type].get_model()

def get_ssl_content(ssl_type, cmodel, wav_16k_tensor):
    """
    Args:
        wav_16k_tensor (torch.Tensor): The audio waveform tensor, with shape [wav_length].
    Returns:
        ssl_content (torch.Tensor): The SSL content tensor, with shape [1, ssl_channels, frame_len].
    """
    assert ssl_type in ssl_map, "ssl type not supported"
    return ssl_map[ssl_type].get_content(cmodel, wav_16k_tensor)








