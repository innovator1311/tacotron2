#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.

# #### Import libraries and setup matplotlib

# In[1]:


import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

from viphoneme import vi2IPA_split
delimit ="/"


hparams = create_hparams()
hparams.sampling_rate = 22050


checkpoint_path = "../drive/MyDrive/MultiSpeaker_Tacotron2/checkpoints/checkpoint_0"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


waveglow_path = '../drive/MyDrive/MultiSpeaker_Tacotron2/checkpoints/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
#denoiser = Denoiser(waveglow)


text = "đây là câu thoại đơn giản"
text = vi2IPA_split(text, delimit)
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()


### 
fpath = Path("../1320_00000.mp3")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
embed = torch.HalfTensor(embed).to("cuda")
###


mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, embed)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))


with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

#audio_denoised = denoiser(audio, strength=0.01)[:, 0]
#ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 

