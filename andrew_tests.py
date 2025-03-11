### SETUP ###
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as pretrained
from models.video_dataset import VideoDataset
from models.srgan import SRGANGenerator, SRGANDiscriminator, SRGANReconstructor
from torch.utils.data import DataLoader

IN_HEIGHT = 2160
IN_WIDTH = 1920
OUT_HEIGHT = 2160
OUT_WIDTH = 3840

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.is_available()

input_path = "data/Seconds_That_Count-checkerboard.mov"
output_path_cam = "data/Seconds_That_Count-cvavg.mov"
output_path_bavg = "data/Seconds_That_Count-bavg.mov"
native_path = "data/Seconds_That_Count-native.mov"
empty_checkerboard_path = "data\LifeSeconds_That_Count-checkerboard-empty.mov"
empty_checkerboard_path = "data/Seconds_That_Count-native.mov"

### SRGAN INITIALIZATION ###
# SRGAN Model
generator = SRGANGenerator().to(device)
discriminator = SRGANDiscriminator(OUT_HEIGHT, OUT_WIDTH).to(device)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)
optimizer_disc = optim.Adam(discriminator.parameters(),lr=1e-3)

# SRGAN Losses
vgg = pretrained.vgg19(weights='VGG19_Weights.DEFAULT').to(device)
gen_loss = nn.BCELoss()
vgg_loss = nn.MSELoss()
mse_loss = nn.MSELoss()
disc_loss = nn.BCELoss()
criterion_generator = nn.MSELoss()

# Dataset
video_dataset = VideoDataset(empty_checkerboard_path, native_path, max_frames=100)
video_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=True)


### SRGAN TRAINING ###
print("Begin Training")
for epoch in range(1):
    adversarial_loss_list = []
    content_loss_list = []
    for input_frames, target_frames in video_dataloader:
        #input_frames = input_frames[:, :OUT_HEIGHT//10, :OUT_WIDTH//10]
        #target_frames = target_frames[:, :OUT_HEIGHT//10, :OUT_WIDTH//10]
        input_frames, target_frames = input_frames.to(device), target_frames.to(device)
    
        discriminator.zero_grad()
        # Generate output frames and use discriminator to predict
        output_frames = generator(input_frames)
        fake_label = discriminator(output_frames)
        real_label = discriminator(target_frames)
        # Update Discriminator
        df_loss = (disc_loss(fake_label,torch.zeros_like(fake_label,dtype=torch.float)))
        dr_loss = (disc_loss(real_label,torch.ones_like(real_label,dtype=torch.float)))
        adversarial_loss = df_loss + dr_loss
        adversarial_loss.backward(retain_graph=True)
        optimizer_disc.step()

        # Update Generator
        generator.zero_grad()  
        g_loss = gen_loss(fake_label.data,torch.ones_like(fake_label,dtype=torch.float))
        v_loss = vgg_loss(vgg.features[:7](output_frames), vgg.features[:7](target_frames))  # VGG 2.2
        m_loss = mse_loss(output_frames, output_frames)
        content_loss = g_loss + v_loss + m_loss
        content_loss.backward(retain_graph=True)
        optimizer_gen.step()
        
        del input_frames, target_frames, output_frames
        torch.cuda.empty_cache()
        adversarial_loss_list.append(adversarial_loss.item())
        content_loss_list.append(content_loss.item())

    print(f"Epoch {epoch+1}, Adversarial Loss: {np.mean(adversarial_loss_list)}, Content Loss: {np.mean(content_loss_list)}")

    # Checkpoint saves
    if (epoch != 0) and (epoch % 5 == 0): 
        torch.save(generator.state_dict(), f"srgan_generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"srgan_discriminator)_{epoch}.pth")

# Save both gen and disc models
torch.save(generator.state_dict(), "srgan_generator.pth")
torch.save(discriminator.state_dict(), "srgan_discriminator.pth")
video_dataset.close()


### RECONSTRUCT WITH SRGAN ###
reconstructor = SRGANReconstructor("srgan_generator.pth")
reconstructor.process_video(empty_checkerboard_path, "output_srgan_reconstructed.mov")
