import torch
import torchvision

class Mel_Classifier(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.patch_size = 14

        self.conv2d1 = torch.nn.Conv2d(384,256, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.relu1 = torch.nn.ReLU()

        self.conv2d2 = torch.nn.Conv2d(256,192, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.relu2 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(768, 264)

    def forward(self, x):
        batch, channel, height, width = x.size()

        if channel == 1:
            x = x.expand(-1, 3, -1, -1)

        next_largest_height_divisor = ((height // self.patch_size) + 1) * self.patch_size
        next_largest_width_divisor = ((width // self.patch_size) + 1) * self.patch_size

        resize_op = torchvision.transforms.Resize(size=(next_largest_height_divisor, next_largest_width_divisor),antialias=True)
        x = resize_op(x)

        # Feed to backbone
        features_dict = self.dinov2_vits14.forward_features(x)
        features = features_dict['x_norm_patchtokens']
        features = features.reshape(batch, 384, next_largest_height_divisor // 14, next_largest_width_divisor // 14)

        features = self.conv2d1(features)
        features = self.pool1(features)
        features = self.relu1(features)

        features = self.conv2d2(features)
        features = self.pool2(features)
        features = self.relu2(features)

        flattened_features = features.view(batch, -1)

        out = self.fc1(flattened_features)

        # x_cls = self.dinov2_vits14(x)
        # x = self.fc1(x_cls)

        return out
    