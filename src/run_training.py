import torch

from config import CONFIG
from dataset import BirdCLEF2023_Dataset
from model import Mel_Classifier
from train import train_net

if __name__ == '__main__':

    # Load Config
    CONFIG = CONFIG()

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

    # Perform Split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (1.0 - CONFIG.val_frac, CONFIG.val_frac))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = CONFIG.batch_size, shuffle = True, num_workers = CONFIG.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = CONFIG.batch_size, shuffle = True, num_workers = CONFIG.num_workers)

    network = Mel_Classifier()
    for param in network.dinov2_vits14.parameters():
        param.requires_grad = False

    criterion = CONFIG.criterion()
    optimizer = CONFIG.optimizer(filter(lambda p: p.requires_grad, network.parameters()), lr=0.0004, momentum=0.9)
    scheduler = CONFIG.scheduler

    train_net(net=network,
              trainloader=train_loader,
              valloader=val_loader,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              epochs=CONFIG.epochs,
              device=CONFIG.device,
              print_every_n_batches=CONFIG.print_every_n_batches,
              outpath=CONFIG.outpath,
              )