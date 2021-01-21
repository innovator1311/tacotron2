import numpy as np
from scipy.io.wavfile import read
import torch

from hparams import create_hparams
hparmas = create_hparams()

from viphoneme import vi2IPA_split
delimit ="/"

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split=" "):

    filepaths_and_text = []
    name = filename.split(".")[0]
    new_file_name = name + "_new.txt"

    with open(new_file_name, 'w') as new_f:
        with open(filename, encoding='utf-8') as f:
            #filepaths_and_text = [line.strip().split(split) for line in f]
            for line in f:
                lines = line.strip().split(split, 1)
                lines[1] = vi2IPA_split(lines[1], delimit)
                
                #lines[1] = hparmas.mel_path + lines[1]
                #lines[2] = hparmas.embed_path + lines[2]
                #print(lines)
                filepaths_and_text.append(lines)
                new_f.write("|".join(lines) + "\n")
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
