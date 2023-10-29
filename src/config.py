import torchvision
import torch


class Config():
    def __init__(self) -> None:

        # Device
        self.device = 'cpu'
        
        # Dataset Path
        self.birdclef2023 = 'birdclef-2023'

        # Out path
        self.outpath = 'results'

        # Train/Validation Split 
        self.val_frac = 0.1

        # Dataloader options
        self.num_workers = 2
        self.batch_size = 4

        # Optimizer Settings
        self.lr=0.0004
        self.momentum=0.9
        self.criterion = torch.nn.CrossEntropyLoss
        self.optimizer = torch.optim.SGD
        self.scheduler = None

        # Training Settings
        self.epochs = 1
        self.print_every_n_batches = 50
        self.patience = 3
        self.fix_features = False

        # Image Transforms
        self.train_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        
        self.val_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        
        self.test_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        # Audio Transforms
        self.train_transforms_audio = None
        
        self.val_transforms_audio = None
        
        self.test_transforms_audio = None


        # Audio Features
        self.sample_rate = 32000
        self.period = 5

        # Mel Spectrogram Parameters
        self.n_fft=2048
        self.f_min=40
        self.f_max=15000
        self.hop_length=512
        self.n_mels=128
        self.mel_args = {'n_fft': self.n_fft,
                         'f_min': self.f_min,
                         'f_max': self.f_max,
                         'hop_length': self.hop_length,
                         'n_mels': self.n_mels}
    