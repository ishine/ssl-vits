from . import contentvec

ssl_map = {
    "contentvec": contentvec
}

def get_ssl_model(ssl_type):
    assert ssl_type in ssl_map, "ssl type not supported"
    return ssl_map[ssl_type].get_model()

def get_ssl_content(ssl_type, cmodel, wav_16k_tensor):
    assert ssl_type in ssl_map, "ssl type not supported"
    return ssl_map[ssl_type].get_content(cmodel, wav_16k_tensor)








