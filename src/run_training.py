import torch
import os
import time

from config import Config
from dataset import BirdCLEF2023_Dataset
from model import Mel_Classifier
from train import train_net, create_logger

def main_train():
    # Load Config
    CONFIG = Config()

    # Change Output path
    folder_name = time.strftime('%Y-%m-%d_%H-%M-%S')
    outpath = os.path.join(CONFIG.outpath, folder_name)
    CONFIG.outpath = outpath
    # Create Output directory
    os.makedirs(CONFIG.outpath, exist_ok=True)

    # Create Logger
    logger = create_logger(final_output_path=CONFIG.outpath)

    # Get all variable to logger
    logger.info('############################################ START CONFIG FILE ############################################')
    for attr, value in vars(CONFIG).items():
        logger.info(f"{attr}: {value}")
    logger.info('############################################  END CONFIG FILE  ############################################')


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

    if CONFIG.fix_features == True:
        for param in network.features.parameters():
            param.requires_grad = False

    criterion = CONFIG.criterion()
    optimizer = CONFIG.optimizer(filter(lambda p: p.requires_grad, network.parameters()), lr=CONFIG.lr, momentum=CONFIG.momentum)
    scheduler = CONFIG.scheduler

    train_net(net=network,
              trainloader=train_loader,
              valloader=val_loader,
              criterion=criterion,
              optimizer=optimizer,
              logging=logger,
              scheduler=scheduler,
              epochs=CONFIG.epochs,
              device=CONFIG.device,
              print_every_samples=CONFIG.print_every_n_batches,
              savePth=CONFIG.outpath,
              patience=CONFIG.patience
              )


if __name__ == '__main__':
    main_train()