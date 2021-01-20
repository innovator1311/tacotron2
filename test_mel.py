from hparams import create_hparams
import layers

from utils import load_wav_to_torch, load_filepaths_and_text

hparams = create_hparams()

stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

audio, sampling_rate = load_wav_to_torch("../1320_00000.mp3")
if sampling_rate != self.stft.sampling_rate:
    raise ValueError("{} {} SR doesn't match target {} SR".format(
        sampling_rate, stft.sampling_rate))
audio_norm = audio / hparams.max_wav_value
audio_norm = audio_norm.unsqueeze(0)
audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
melspec = stft.mel_spectrogram(audio_norm)
melspec = torch.squeeze(melspec, 0)

print(melspec.shape)