import torch
import os
import pickle

from model import DeepJSCC
from alignment.alignment_model import AlignedDeepJSCC, _LinearAlignment

def load_deep_jscc(path, snr, c, channel):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v

    model = DeepJSCC(c=c, channel_type=channel, snr=snr)

    model.load_state_dict(new_state_dict)
    model.change_channel(channel, snr)

    return model

def load_aligned_model(model1_fp, model2_fp, aligner_fp, snr, c, channel):

    model1 = load_deep_jscc(model1_fp, snr, c, channel)
    model2 = load_deep_jscc(model2_fp, snr, c, channel)

    if aligner_fp is None:
        aligner = None
    
    else:
        with open(aligner_fp, 'rb') as f:
            align_matrix = pickle.load(f)

        aligner = _LinearAlignment(align_matrix)

    return AlignedDeepJSCC(model1, model2, aligner)