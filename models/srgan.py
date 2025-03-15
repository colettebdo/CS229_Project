import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np
import os

class _ResidualBlock(nn.Module):
    def __init__(self, channels=3):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.prelu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        return x + out  # Residual connection

class _UpscaleBlock(nn.Module):
    def __init__(self):
        super(_UpscaleBlock, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

class SRGANGenerator(nn.Module):
    def __init__(self, n_residual_blocks=1, upscale_factor=2):
        super(SRGANGenerator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=9, padding=4, bias=False),
            nn.PReLU()
        )
        self.b_blocks = nn.Sequential(*[_ResidualBlock(48) for _ in range(n_residual_blocks)])
        self.upscale_blocks = nn.Sequential(*[_UpscaleBlock() for _ in range(int(np.log2(upscale_factor)))]) if upscale_factor > 1 else None
        self.final_conv = nn.Conv2d(48, 3, kernel_size=5, padding=2, bias=False)

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
        
        # Single convolutional layer with downsampling
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, stride=2, padding=1, bias=False),  # Downsampling
            nn.LeakyReLU(0.2, True),
        )

        # Compute FC input size dynamically
        lc_input_len = self._get_linear_input_size(img_H, img_W, n_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(lc_input_len, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)  # Apply single conv layer
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.classifier(x)  # Binary classification (real vs fake)
        return x
    
    def _get_linear_input_size(self, H, W, channels):
        """Compute the input size of the final FC layer after a single conv layer."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, H, W)  # Simulate input
            x = self.conv(dummy_input)
            return x.shape[1] * x.shape[2] * x.shape[3]  # Corrected flatten size



class SRGANReconstructor:
    def __init__(self, gen_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = SRGANGenerator().to(self.device)
        self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
        self.generator.eval()

    def _reconstruct_frame(self, frame):
        """Upscale a single frame from 1080p to 4K"""
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        frame = frame.to(self.device)

        with torch.no_grad():
            output = self.generator(frame).squeeze(0)

        output = (output * 255.0).permute(1, 2, 0).cpu().numpy()
        return np.clip(output, 0, 255).astype(np.uint8)  # Clip before conversion

        return (output.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

    def process_video(self, input_path, output_path, max_frames=None, batch_size=1):
        """Processes an entire video from 1080p → 4K"""
        cap = cv.VideoCapture(input_path)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        total_frames = max_frames if max_frames else int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * 2
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2

        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))  # Output at 4K

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            reconstructed_frame = self._reconstruct_frame(frame)
            # upscaled_frame_np = cv.cvtColor(upscaled_frame_np, cv.COLOR_BGR2HSV)
            out.write(reconstructed_frame)

        cap.release()
        out.release()
        print(f"Reconstructed 4K video saved to {output_path}")

    def process_image(self, input_image_path):
        """
        Takes an image path as input, processes it using the reconstruct_frame method, 
        and saves the reconstructed image with a modified filename.
        
        Args:
            input_image_path (str): Path to the input image file.
        
        Returns:
            str: Path to the saved reconstructed image.
        """
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        # Load the image
        input_image = cv.imread(input_image_path)
        if input_image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        # Process the image
        reconstructed_image = self._reconstruct_frame(input_image)

        # Generate output image path
        directory, filename = os.path.split(input_image_path)
        name, ext = os.path.splitext(filename)
        output_image_path = os.path.join(directory, f"{name}_reconstructed_srgan{ext}")

        # Save the processed image
        cv.imwrite(output_image_path, reconstructed_image)

        print(f"Reconstructed image saved to {output_image_path}")
        return output_image_path

