'''
Architecture reference: https://arxiv.org/abs/1609.04802
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np

class _ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.prelu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        return x + out      # Residual connection
    
class _UpscaleBlock(nn.Module):
    '''This block upscales an image of shape (H,W,C) to (H*2, W*2, C)'''
    def __init__(self, channels=64):
        super(_UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels*4, kernel_size=3, padding=1, bias=False)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelshuffle(x)
        x = self.prelu(x)
        return x


class SRGANGenerator(nn.Module):
    def __init__(self, n_residual_blocks=16, upscale_factor=1):
        super(SRGANGenerator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=False),
            nn.PReLU() 
        )
        self.b_blocks = nn.Sequential(*[_ResidualBlock(64) for _ in range(n_residual_blocks)])
        self.upscale_blocks = nn.Sequential(*[_UpscaleBlock(64) for _ in range(int(np.log2(upscale_factor)))]) if upscale_factor > 1 else None
        self.final_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=False)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.b_blocks(x)
        if self.upscale_blocks:
            x = self.upscale_blocks(x)
        x = self.final_conv(x)
        return x


class SRGANDiscriminator(nn.Module):
    def __init__(self, img_H, img_W, n_channels=64):
        super(SRGANDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_channels*2, n_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_channels*4, n_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*4),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_channels*4, n_channels*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_channels*8, n_channels*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_channels*8),
            nn.LeakyReLU(0.2, True),
        )
        lc_input_len = self._get_linear_input_size(img_H, img_W)
        self.classifier = nn.Sequential(
            nn.Linear(lc_input_len, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        print('disc final layer: ', x.shape)
        return x
    
    def _get_linear_input_size(self, H, W):
        dummy_input = torch.randn(1, 3, H, W)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        print('flatten shape:', x.shape )
        return x.shape[1]




class SRGANReconstructor:
    def __init__(self, gen_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = SRGANGenerator().to(self.device)
        self.generator.load_state_dict(torch.load(gen_path))
        self.generator.eval()
        ''' 
        self.discriminator = SRGANDiscriminator(..., ...).to(self.device)
        self.discriminator.load_state_dict(torch.load(disc_path))
        self.discriminator.eval()
        '''

    def _reconstruct_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # / 255.0
        frame = frame.to(self.device)
        # print(frame.shape)
        # return (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        with torch.no_grad():
            output = self.generator(frame).squeeze(0)
        return (output.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)


    def process_video(self, input_path, output_path):
        cap = cv.VideoCapture(input_path)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        i = 0
        while cap.isOpened() and i < 500:
            ret, frame = cap.read()
            if not ret:
                break

            reconstructed_frame = self._reconstruct_frame(frame)
            upscaled_frame_np = np.clip(reconstructed_frame, 0, 255).astype(np.uint8)

            upscaled_frame_np = cv.cvtColor(upscaled_frame_np, cv.COLOR_RGB2BGR)

            out.write(upscaled_frame_np)
            i+=1

        cap.release()
        out.release()
        print(f"Reconstructed video saved to {output_path}")
