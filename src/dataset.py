import os

import torch
import torchaudio
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def load_wave_and_crop(filename, period, start=None):

    waveform_orig, sample_rate = torchaudio.load(filename)

    wave_len = len(waveform_orig)
    waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])

    effective_length = sample_rate * period
    while len(waveform[0,:]) < (period * sample_rate * 3):
        waveform = np.concatenate([waveform, waveform_orig])
    if start is not None:
        start = start - (period - 5) / 2 * sample_rate
        while start < 0:
            start += wave_len
        start = int(start)
    else:
        if wave_len < effective_length:
            start = np.random.randint(effective_length - wave_len)
        elif wave_len > effective_length:
            start = np.random.randint(wave_len - effective_length)
        elif wave_len == effective_length:
            start = 0

    waveform_seg = waveform[start: start + int(effective_length)]

    return waveform_orig, waveform_seg, sample_rate, start

class BirdCLEF2023_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, sample_rate: float = 32000, n_fft=2048, f_min = 20, f_max = 16000 ,hop_length=512, n_mels=128, wave_transform = None, period = 5):

        # Save path of dataset
        self.datapath = data_path

        # Get metadata
        self.df = pd.read_csv(os.path.join(data_path, 'train_metadata.csv'))

        # Save hyperparameters
        self.sample_rate = sample_rate
        self.period = period

        # Initialize Mel Spectrogram Object
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,  
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            hop_length=hop_length,      
            n_mels=n_mels,           
        )

        # Initialize Transform object
        self.wave_transform = wave_transform

        # Get species list
        self.species = list(set(self.df['primary_label']))
        
        return
    
    def __len__(self):
        length = len(list(self.df['primary_label']))
        return length

    def __getitem__(self, idx):

        dict_idx = dict(self.df.iloc[idx])

        # Get labels as torch tensors
        primary_label = torch.tensor([1 if dict_idx['primary_label'] == label else 0 for label in self.species],dtype=float)
        secondary_label = torch.tensor([1 if dict_idx['secondary_labels'] == label else 0 for label in self.species],dtype=float)
        dict_idx['primary_label_tensor'] = primary_label
        dict_idx['secondary_label_tensor'] = secondary_label

        # # Get path of wave file and load
        # ogg_file = os.path.join(self.datapath, os.path.join('train_audio',dict_idx['filename']))
        # waveform, sample_rate = torchaudio.load(ogg_file)

        # Load and Crop
        ogg_file = os.path.join(self.datapath, os.path.join('train_audio',dict_idx['filename']))
        waveform, waveform_seg, sample_rate, start = load_wave_and_crop(ogg_file, self.period)
        # waveform_seg = self.wave_transforms(samples=waveform_seg, sample_rate=self.sample_rate)

        # Resampling to a target sample rate
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        waveform = resampler(waveform)
        dict_idx['waveform'] = waveform

        # Get mel spectrogram
        mel_spectrogram = self.mel_transform(waveform)
        dict_idx['mel_spec'] = mel_spectrogram

        # return dict_idx
        return mel_spectrogram, primary_label

if __name__=="__main__":
    dataset = BirdCLEF2023_Dataset(data_path = 'birdclef-2023'
                                   ,sample_rate = 32000)
    data_dict = dataset[0]
    mel_spec = data_dict['mel_spec']
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.show()
    