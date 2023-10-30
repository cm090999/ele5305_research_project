
# https://www.kaggle.com/code/nischaydnk/split-creating-melspecs-stage-1/notebook
import torch.nn as nn
import torchaudio
import pandas as pd
from torchvision.utils import save_image
import numpy as np
from PIL import Image

import os
from tqdm import tqdm

from config import Config_Mel

class MelSpectrogram_Builder():
    def __init__(self, device, **mel_kwargs) -> None:
        self.mel_kwargs = mel_kwargs
        self.device = device

        # Initialize Mel Spectrogram Object
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            **self.mel_kwargs
        )
        self.mel_transform.to(self.device)
        return
    
    def __call__(self, waveform):
        
        # Compute the Mel spectrogram
        mel_spectrogram = self.mel_transform(waveform.to(self.device))
        return mel_spectrogram

class BirdClef_ToMel(nn.Module):
    def __init__(self, 
                 datapath,
                 melspecBuilder,
                 melSpecTransform,
                 outpath,
                 **kwargs) -> None:
        super().__init__()

        # Default values
        self.sample_rate = kwargs.get('sample_rate', 32000)

        # Save path of dataset
        self.datapath = datapath
        self.outpath = outpath

        # Transform
        self.mel_spec_transform = melSpecTransform

        # Get metadata
        self.df = pd.read_csv(os.path.join(datapath, 'train_metadata.csv'))

        # Get Mel Spectrogram Builder Object
        self.melspecBuilder = melspecBuilder

        # Get device
        self.device = self.melspecBuilder.device

        # Get species list
        self.species = list(set(self.df['primary_label']))

        return

    def __len__(self):
        length = len(list(self.df['primary_label']))
        return length
    
    def __getitem__(self, idx):

        dict_idx = dict(self.df.iloc[idx])

        return dict_idx
    
    def create_directory(self):

        imagePath = os.path.join(self.outpath, 'images')
        os.makedirs(imagePath, exist_ok=True)
        
        return
    
    def compute_mel_spec(self, idx):

        ogg_file = os.path.join(self.datapath, os.path.join('train_audio',dict(self.df.iloc[idx])['filename']))
        waveform, sample_rate = torchaudio.load(ogg_file)
        waveform = waveform.ravel()

        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=self.sample_rate)
        waveform = resampler(waveform)

        mel_spectrogram = self.melspecBuilder(waveform)
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1)
        if self.mel_spec_transform is not None:
            mel_spectrogram = self.mel_spec_transform(mel_spectrogram)
        
        return mel_spectrogram
    
    def normalized_to_uint8(self,tensorImage):
        tensorImage = (tensorImage - tensorImage.min()) / (tensorImage.max() - tensorImage.min())
        numpyImage = self.toNumpy(tensorImage=tensorImage)
        numpyImage = (numpyImage * 255).astype(np.uint8)
        return numpyImage
    
    def toNumpy(self, tensorImage):
        tensorImage = tensorImage.cpu().detach().numpy()
        return tensorImage
    
    def saveMelSpec(self,image, name):
        image.save(name)
        return

    def createDataset(self):

        # Create empty directory
        self.create_directory()

        # List to store image paths
        image_paths = [None] * len(self)

        # Loop over dataset
        for i in tqdm(range(len(self)), desc="Processing"):
            # Calculate mel spectrogram
            mel_spec = self.compute_mel_spec(i)

            # Convert spectrogram to numpy
            mel_spec = self.normalized_to_uint8(mel_spec)

            # Convert to PIL
            mel_spec = mel_spec.transpose(1, 2, 0)
            mel_spec = Image.fromarray(mel_spec)

            # Save name
            image_name = str(i).zfill(6) + '.png'
            savepath_ = os.path.join(self.outpath, 'images')
            savepath = os.path.join(savepath_,image_name)

            # Save image
            self.saveMelSpec(image=mel_spec, name=savepath)

            # TODO: Write path to df
            relative_path = os.path.relpath(savepath,self.outpath)
            image_paths[i] = relative_path

        # TODO: Save df as metadata
        # Add a new column 'image_path' to the DataFrame
        self.df['image_path'] = image_paths

        # TODO: Save df as .csv
        csv_path = os.path.join(self.outpath, 'train_metadata.csv')
        self.df.to_csv(csv_path, index=False)

        return




if __name__=='__main__':

    CONFIG = Config_Mel()

    mel_builder = MelSpectrogram_Builder(device=CONFIG.device, **CONFIG.mel_args)
    image_builder = BirdClef_ToMel(datapath=CONFIG.birdclef2023,
                                   melspecBuilder=mel_builder,
                                   melSpecTransform=None,
                                   outpath=CONFIG.outpath_images)
    image_builder.createDataset()
    pass