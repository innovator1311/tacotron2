import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        
        #print("Enter loader")

        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.hparams = hparams

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, embed, text = self.hparams.mel_path + audiopath_and_text[0] + ".npy", self.hparams.embed_path + audiopath_and_text[0] + ".npy", audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        embed = torch.from_numpy(np.load(embed))
        
        #return (text, mel, embed)
        return (text, mel, embed)

    def get_mel(self, filename):

        if not self.load_mel_from_disk:

            full_path = self.hparams.mel_path + filename
            #full_path = "normalized/" + filename + ".wav"

            audio, sampling_rate = load_wav_to_torch(full_path)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)

            #Do the saving
            
            np.save("../vinbigdata_preprocess/mels/{}".format(filename), melspec)

            fpath = Path(full_path)
            wav = preprocess_wav(fpath)

            encoder = VoiceEncoder()
            embed = encoder.embed_utterance(wav)
            np.save("../vivos_preprocess/embeds/{}".format(filename), embed)
            
            ##

            melspec = torch.squeeze(melspec, 0)
        else:
            full_path = filename
            #print("File name, ", filename)
            melspec = torch.from_numpy(np.load(full_path))
            melspec = melspec.squeeze()
            #melspec = melspec[:,:80]

            
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        #print(max_target_len)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded # and embed
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        embed_tensor = torch.FloatTensor(len(batch), 256)

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            embedd = batch[ids_sorted_decreasing[i]][2]

            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            embed_tensor[i, :256] = embedd 
        

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, embed_tensor

import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        
        #print("Enter loader")

        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.hparams = hparams

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, embed, text = self.hparams.mel_path + audiopath_and_text[0] + ".npy", self.hparams.embed_path + audiopath_and_text[0] + ".npy", audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        embed = torch.from_numpy(np.load(embed))
        
        #return (text, mel, embed)
        return (text, mel, embed)

    def get_mel(self, filename):

        if not self.load_mel_from_disk:

            full_path = self.hparams.mel_path + filename
            #full_path = "normalized/" + filename + ".wav"

            audio, sampling_rate = load_wav_to_torch(full_path)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)

            #Do the saving
            
            np.save("../vinbigdata_preprocess/mels/{}".format(filename), melspec)

            fpath = Path(full_path)
            wav = preprocess_wav(fpath)

            encoder = VoiceEncoder()
            embed = encoder.embed_utterance(wav)
            np.save("../vivos_preprocess/embeds/{}".format(filename), embed)
            
            ##

            melspec = torch.squeeze(melspec, 0)
        else:
            full_path = filename
            #print("File name, ", filename)
            melspec = torch.from_numpy(np.load(full_path))
            melspec = melspec.squeeze()
            #melspec = melspec[:,:80]

            
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class NewTextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        #print(max_target_len)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded # and embed
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        embed_tensor = torch.FloatTensor(len(batch), 256)

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            embedd = batch[ids_sorted_decreasing[i]][2]

            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            embed_tensor[i, :256] = embedd 
        

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, embed_tensor

class NewTextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        
        #print("Enter loader")

        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.hparams = hparams
        self.encoder = VoiceEncoder()

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, embed, text = self.hparams.mel_path + audiopath_and_text[0] + ".npy", self.hparams.embed_path + audiopath_and_text[0] + ".npy", audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        embed = torch.from_numpy(np.load(embed))
        
        #return (text, mel, embed)
        return (text, mel, embed)

    def get_mel(self, filename):

        #full_path = self.hparams.mel_path + filename
        #full_path = "normalized/" + filename + ".wav"

        audio, sampling_rate = load_wav_to_torch(full_path)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)

        #Do the saving
        np.save("../vinbigdata_preprocess/mels/{}".format(filename), melspec)

        fpath = Path(full_path)
        wav = preprocess_wav(fpath)

        embed = self.encoder.embed_utterance(wav)
        np.save("../vinbigdata_preprocess/embeds/{}".format(filename), embed)
        ##

        melspec = torch.squeeze(melspec, 0)
        
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

