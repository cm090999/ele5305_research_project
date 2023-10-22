import torch
import torchvision

class Mel_Classifier(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.patch_size = 14
        self.fc1 = torch.nn.Linear(384, 264)

    def forward(self, x):
        batch, channel, height, width = x.size()

        if channel == 1:
            x = x.expand(-1, 3, -1, -1)

        next_largest_height_divisor = ((height // self.patch_size) + 1) * self.patch_size
        next_largest_width_divisor = ((width // self.patch_size) + 1) * self.patch_size

        resize_op = torchvision.transforms.Resize(size=(next_largest_height_divisor, next_largest_width_divisor),antialias=True)
        x = resize_op(x)
        x = self.dinov2_vits14(x)
        x = self.fc1(x)

        return x