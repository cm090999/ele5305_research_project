import os
from typing import Any

import torch
import torchaudio
import torchvision
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def load_wave_and_crop(filename, period, start=None):

    waveform_orig, sample_rate = torchaudio.load(filename)
    waveform_orig = waveform_orig.ravel()

    wave_len = len(waveform_orig)
    waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])

    effective_length = sample_rate * period
    while len(waveform) < (period * sample_rate * 3):
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
    def __init__(self, data_path: str, **kwargs):

        # Default values
        sample_rate = kwargs.get('sample_rate', 32000)
        n_fft = kwargs.get('n_fft', 2048)
        f_min = kwargs.get('f_min', 40)
        f_max = kwargs.get('f_max', 15000)
        hop_length = kwargs.get('hop_length', 512)
        n_mels = kwargs.get('n_mels', 128)
        wave_transform = kwargs.get('wave_transform', None)
        period = kwargs.get('period', 5)
        secondary_coef = kwargs.get('secondary_coef', 1.0)
        smooth_label = kwargs.get('smooth_label', 0.05)
        mel_spec_transform = kwargs.get('mel_spec_transform', None)
        n_mfcc = kwargs.get('n_mfcc', 40)
        device = kwargs.get('device', 'cpu')
         
        # Save path of dataset
        self.datapath = data_path

        # Get metadata
        self.df = pd.read_csv(os.path.join(data_path, 'train_metadata.csv'))

        # Save hyperparameters
        self.device = device
        self.sample_rate = sample_rate
        self.period = period
        self.secondary_coef = secondary_coef
        self.smooth_label = smooth_label
        self.n_mfcc = n_mfcc

        # Initialize Mel Spectrogram Object
        mel_kwargs = {  'n_fft': n_fft,
                        'f_min': f_min,
                        'f_max': f_max,
                        'hop_length': hop_length,
                        'n_mels': n_mels 
                     }
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            **mel_kwargs          
        )

        # Initialize MFCC extractor
        self.mfcc_extractor = torchaudio.transforms.MFCC(n_mfcc=self.n_mfcc,
                                                         sample_rate=sample_rate,
                                                        melkwargs=mel_kwargs)
        
        # # Send to device
        # self.mel_transform.to(device)
        # self.mfcc_extractor.to(device)

        # Initialize Transform object
        self.wave_transform = wave_transform
        self.mel_spec_transform = mel_spec_transform

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
        secondary_label = torch.tensor([1 if label in dict_idx['secondary_labels'] else 0 for label in self.species], dtype=float)
        combined_label = self._prepare_target(main_tgt=primary_label, sec_tgt=secondary_label)
        dict_idx['combined_label_tensor'] = combined_label
        dict_idx['primary_label_tensor'] = primary_label
        dict_idx['secondary_label_tensor'] = secondary_label

        # Load and Crop
        ogg_file = os.path.join(self.datapath, os.path.join('train_audio',dict_idx['filename']))
        waveform, waveform_seg, sample_rate, start = load_wave_and_crop(ogg_file, self.period)
        waveform_seg = torch.tensor(waveform_seg)
        # waveform_seg = self.wave_transforms(samples=waveform_seg, sample_rate=self.sample_rate)

        # Resampling to a target sample rate
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        waveform_seg = resampler(waveform_seg)
        dict_idx['waveform_seg'] = waveform_seg

        # Get mel spectrogram
        mel_spectrogram = self.mel_transform(waveform_seg)
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1)
        if self.mel_spec_transform is not None:
            mel_spectrogram = self.mel_spec_transform(mel_spectrogram)
        dict_idx['mel_spec'] = mel_spectrogram

        # Get MFCCs
        mfcc = self.mfcc_extractor(waveform_seg)
        dict_idx['mfcc'] = mfcc

        # return dict_idx
        return dict_idx
    
    # https://github.com/VSydorskyy/BirdCLEF_2023_1st_place/blob/main/code_base/datasets/wave_dataset.py, changed
    def _prepare_target(self, main_tgt, sec_tgt, all_labels=None):
        all_tgt = main_tgt + sec_tgt
        all_tgt = torch.clamp(all_tgt, 0.0, 1.0)
        return all_tgt

if __name__=="__main__":

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    dataset = BirdCLEF2023_Dataset(data_path = 'birdclef-2023'
                                   ,sample_rate = 32000,
                                   mel_spec_transform=transforms)
    
    data_dict = dataset[5]
    mel_spec = data_dict['mel_spec']

    # mel_builder = MelSpectrogram_Builder(device='cpu', sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128)

    # Call the instance with a waveform to compute the Mel spectrogram
    # mel_spectrogram = mel_builder(data_dict['waveform_seg'])

    mfcc = data_dict['mfcc']
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.show()


# TODO: outsource operations to gpu?
    