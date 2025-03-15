"""
Architecture Reference: https://arxiv.org/pdf/1608.00367v1
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np
from models.video_dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import copy
import torchvision.models as models
from torchvision import transforms
import os


"""
We intended to use VGGPerceptualLoss for the FSRCNN, however, it posed memory issues and affected
our eventual predicted frames. 

TODO: Figure out how VGG could be useful in improving optimization
"""
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer=16):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:layer])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        generated_features = self.vgg_layers(generated)
        target_features = self.vgg_layers(target)
        loss = torch.nn.functional.mse_loss(generated_features, target_features)
        return loss

class SuperResCNN(nn.Module):
    def __init__(self, scale=2, channels=3, feature_dim_d=64, shrinking_filters_s=12):
        super(SuperResCNN, self).__init__()

        self.feature_extraction_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=feature_dim_d,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(feature_dim_d),  # Added Batch Normalization for Color
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.shrinking_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_dim_d,
                out_channels=shrinking_filters_s,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(shrinking_filters_s),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.nonlinear_mapping_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=shrinking_filters_s, out_channels=shrinking_filters_s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(shrinking_filters_s),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(
                in_channels=shrinking_filters_s, out_channels=shrinking_filters_s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(shrinking_filters_s),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(
                in_channels=shrinking_filters_s, out_channels=shrinking_filters_s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(shrinking_filters_s),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(
                in_channels=shrinking_filters_s, out_channels=shrinking_filters_s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(shrinking_filters_s),
            nn.LeakyReLU(negative_slope=0.2),
            
        )

        self.expanding_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=shrinking_filters_s,
                out_channels=feature_dim_d,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(feature_dim_d), 
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.deconvolution_layer = nn.ConvTranspose2d(
            in_channels=feature_dim_d,
            out_channels=channels,
            kernel_size=9,
            stride=scale,
            padding=4,
            output_padding=scale - 1
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Feature Extraction layer initialization
        module = self.feature_extraction_layer[0]
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(module.bias.data)
        
        # Shrinking Layer initialization
        module = self.shrinking_layer[0]
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(module.bias.data)

        # Nonlinear Mapping layer initialization
        for module in self.nonlinear_mapping_layer:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias.data)

        # Expanding layer initialization
        module = self.expanding_layer[0]
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(module.bias.data)

        # Deconvolution layer initialization
        nn.init.kaiming_normal_(self.deconvolution_layer.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.deconvolution_layer.bias.data)

    def forward(self, x):
        x = self.feature_extraction_layer(x)
        x = self.shrinking_layer(x)
        x = self.nonlinear_mapping_layer(x)
        x = self.expanding_layer(x)
        x = self.deconvolution_layer(x)
        return x

class CNNReconstructor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SuperResCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def reconstruct_frame(self, frame):
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        frame = frame.to(self.device)

        with torch.no_grad():
            output = self.model(frame).squeeze(0)

        output = (output * 255.0).permute(1, 2, 0).cpu().numpy()
        return np.clip(output, 0, 255).astype(np.uint8)  # Clip before conversion
        # normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # output = normalized(output)
        return output

    def process_video(self, input_path, output_path):
        cap = cv.VideoCapture(input_path)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * 2
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        print(width, height)
        i = 0
        while cap.isOpened() and i < 500:
            ret, frame = cap.read()
            if not ret:
                break

            reconstructed_frame = self.reconstruct_frame(frame)
            # upscaled_frame_np = cv.cvtColor(upscaled_frame_np, cv.COLOR_BGR2HSV)
            out.write(reconstructed_frame)
            i+=1

        cap.release()
        out.release()
        print(f"Reconstructed video saved to {output_path}")
    

    def process_image(self, input_image_path):
        input_image = cv.imread(input_image_path)
        if input_image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        reconstructed_image = self.reconstruct_frame(input_image)
        directory, filename = os.path.split(input_image_path)
        name, ext = os.path.splitext(filename)
        output_image_path = os.path.join(directory, f"{name}_reconstructed{ext}")

        cv.imwrite(output_image_path, reconstructed_image)

        print(f"Reconstructed image saved to {output_image_path}")
        return output_image_path



def peak_signal_noise(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    return 10. * torch.log10(1. / (mse + eps))


def trainSuperResCNN(training, test, epoches, lr, max_frames, pretrained=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperResCNN().to(device)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint)
    
    optimizer = optim.Adam([
        {'params': model.feature_extraction_layer.parameters()},
        {'params': model.shrinking_layer.parameters()},
        {'params': model.nonlinear_mapping_layer.parameters()},
        {'params': model.expanding_layer.parameters()},
        {'params': model.deconvolution_layer.parameters()}
    ], lr=lr)

    mse_loss = nn.MSELoss()
    # vgg_loss = VGGPerceptualLoss().cuda()
    lda = 0.1
    
    def loss_function(output, target):
        mse = mse_loss(output, target)
        return mse
        # perceptual = vgg_loss(output, target)
        total_loss = mse + lda * perceptual
        return total_loss

    model.train()

    stride = int(max_frames / 10.0)
    video_dataset = VideoDataset(training, test, max_frames=max_frames, stride=stride)
    video_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=True)

    final_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    final_psnr_avg = -2**10
    final_loss = 0

    print("Begin Training")
    for epoch in range(epoches):
        video_dataset.set_epoch(epoch % stride)
        model.train()
        epoch_loss_sum, epoch_loss_count = 0, 0
        epoch_psnr_sum, epoch_psnr_count = 0, 0

        for input, target in video_dataloader:
            input, target = input.to(device), target.to(device)
            predicted = model(input)
            predicted = torch.clamp(predicted, 0, 1)
            loss = loss_function(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * len(input)
            epoch_loss_count += len(input)
            epoch_psnr_sum += peak_signal_noise(predicted, target) * len(input)
            epoch_psnr_count += len(input)
            # print(epoch_loss_count)

            del input, target, predicted
            torch.cuda.empty_cache()

        if epoch % stride == 0:
            torch.save(model.state_dict(), 'srcnn_epoch_{}.pth'.format(epoch // stride))

        epoch_loss_avg = epoch_loss_sum / epoch_loss_count
        epoch_psnr_avg = epoch_psnr_sum / epoch_psnr_count

        print(f"Epoch {epoch}, Loss: {epoch_loss_avg}, PSNR: {epoch_psnr_avg}")

        if epoch_psnr_avg > final_psnr_avg:
            best_epoch = epoch
            final_psnr_avg = epoch_psnr_avg
            final_weights = copy.deepcopy(model.state_dict())
            final_loss = epoch_loss_avg

    print(f"Best Epoch {best_epoch}, Loss: {final_loss}, PSNR: {final_psnr_avg}")
    torch.save(final_weights, 'srcnn_best.pth')
    video_dataset.close()

