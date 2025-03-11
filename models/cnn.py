import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np

class CNNInterpolator(nn.Module):
    def __init__(self):
        super(CNNInterpolator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNNReconstructor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNInterpolator().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def reconstruct_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)# / 255.0
        frame = frame.to(self.device)
        # print(frame.shape)
        # return (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        with torch.no_grad():
            output = self.model(frame).squeeze(0)
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

            reconstructed_frame = self.reconstruct_frame(frame)
            upscaled_frame_np = np.clip(reconstructed_frame, 0, 255).astype(np.uint8)

            upscaled_frame_np = cv.cvtColor(upscaled_frame_np, cv.COLOR_RGB2BGR)

            out.write(upscaled_frame_np)
            i+=1

        cap.release()
        out.release()
        print(f"Reconstructed video saved to {output_path}")
