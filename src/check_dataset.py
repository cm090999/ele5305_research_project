from dataset import BirdCLEF2023_Dataset
from config import Config

import matplotlib.pyplot as plt
import random
from IPython.display import Audio

def main_test_dataset():


    # Load Config
    CONFIG = Config()

    # Get Dataset
    dataset = BirdCLEF2023_Dataset(data_path=CONFIG.birdclef2023,
                                   sample_rate=CONFIG.sample_rate,
                                   n_fft=CONFIG.n_fft,
                                   f_min=CONFIG.f_min,
                                   f_max=CONFIG.f_max,
                                   hop_length=CONFIG.hop_length,
                                   n_mels=CONFIG.n_mels,
                                   wave_transform=None,
                                   mel_spec_transform=None,
                                   period=CONFIG.period)
    plt.ioff()
    while True:
        rand_idx = random.randint(0, len(dataset) - 1)
        data_dict = dataset[rand_idx]
        data_label = data_dict['primary_label']
        mel_spec = data_dict['mel_spec']
        audio_file = data_dict['waveform_seg']

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
        plt.title("Mel Spectrogram {}".format(data_label))
        plt.colorbar(format="%+2.0f dB")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        plt.show()

        Audio(data=audio_file.numpy(), rate=32000)

if __name__=='__main__':
    main_test_dataset()