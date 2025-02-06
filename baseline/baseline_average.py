import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

IN_HEIGHT = 2160
IN_WIDTH = 1920
OUT_HEIGHT = 2160
OUT_WIDTH = 3840

class BaselineAverageModel(nn.Module):
    def __init__(self, in_shape=(IN_HEIGHT, IN_WIDTH), out_shape=(OUT_HEIGHT, OUT_WIDTH)):
        super(BaselineAverageModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, x):
        '''
        Given a downscaled video x, return an upscaled version of it y. TODO: Revise this into a parallel, non-iterative approach.

        Args:
            x: a tensor of shape (N_TimeFrames, IN_HEIGHT, IN_WIDTH, N_Pixel_Channels)

        Returns:
            y: a tensor of shape (N_TimeFrames, OUT_HEIGHT, OUT_WIDTH, N_Pixel_Channels)
        '''
        T, H, W, C = x.shape
        # Convert video to float32 for processing
        x = x.to(torch.float32)
        
        # Construct upscaled video frame-by-frame
        y = torch.zeros((T, H, 2*W, C), dtype=torch.float32)
        for t in range(T):
            in_frame = x[t]    # shape of (H, W, C)
            # average color of the whole frame
            avg_color = torch.mean(in_frame, dim=(0, 1), dtype=torch.float32)

            # output frame
            out_frame = torch.zeros((H, 2*W, C))
            # fill in original pixels in checkerboard manner
            out_frame[::2, ::2] = in_frame[::2]
            out_frame[1::2, 1::2] = in_frame[1::2]
            # fill in empty gaps with the average color
            out_frame[::2, 1::2] = avg_color
            out_frame[1::2, ::2] = avg_color
            y[t] = out_frame
        # Convert 
        y = y.to(torch.uint8)
        return y
    
        T, H, W, C = x.shape
        # Convert video to float32 for processing
        x = x.to(torch.float32)

        # Compute average color for each frame in batch 
        avg_colors = torch.mean(x, dim=(1,2))  # shape: (T, C)
        print(avg_colors.shape)

        # Initialize output tensor
        y = torch.zeros((T, H, 2 * W, C), dtype=torch.float32)

        # Fill original pixels into expanded positions
        y[:, ::2, ::2] = x       # Original pixels at even rows/columns
        y[:, 1::2, 1::2] = x     # Original pixels at odd rows/columns

        # Fill in original pixels in checkerboard manner
        y[:, ::2, 1::2] = avg_colors[:]
        # Fill in empty gaps with the average color of that frame
        y[:, 1::2, ::2] = avg_colors[:]
        return y


# TODO: Put to another file later
def mp4_to_tensor(video_path, startpoint=0.0, endpoint=None):
    '''
    Converts an mp4 into a PyTorch Tensor. 
    Args:
        video_path [string]: path to the mp4 video that we want to convert
        startpoint [float/fraction]: timestamp (in seconds) of where to start
        endpoint [float/fraction]: timestamp (in seconds) of where to end
    Returns:
        video: a Tensor of shape (N_TimeFrames, Height, Width, N_Pixel_Channel)
    '''
    video, _, _ = torchvision.io.read_video(video_path, start_pts=startpoint, end_pts=endpoint, pts_unit='sec')
    return video

def tensor_to_mp4(video_tensor, save_path, fps=24.0):
    '''
    Converts a PyTorch Tensor into an mp4 file. 
    Args:
        video_tensor: a Tensor of shape (N_TimeFrames, Height, Width, N_Pixel_Channel); the tensor we want to convert
        save_path [string]: path to save the mp4 video that we create
        fps [float]: frames per second for the video
    Returns:
        Nothing
    '''
    T, H, W, C = video_tensor.shape
    video_np = (video_tensor.cpu().detach()).numpy()    # convert to np format so cv2 works
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    
    for t in range(T):
        frame = video_np[t] 
        video_writer.write(frame) 

    video_writer.release()
    print(f"Video saved to {save_path}")


def main():
    # Read in video data and ensure its sizing is correct
    input_path = "data/Seconds_That_Count-small.mp4"
    input_frames = mp4_to_tensor(input_path, startpoint=0.2, endpoint=1.0)
    if (input_frames.shape[1:3] != (IN_HEIGHT, IN_WIDTH)):
        raise ValueError(f'Input video of frame size {tuple(input_frames.shape[1:3])} does not match the expected size of ({IN_HEIGHT}, {IN_WIDTH}).')
    
    # Instantiate model
    model = BaselineAverageModel(in_shape=(IN_HEIGHT, IN_WIDTH), out_shape=(OUT_HEIGHT, OUT_WIDTH))

    # Upscale input video
    output_frames = model(input_frames)
    print(f"Upscaled input to tensor of shape {tuple(output_frames.shape)}")
    output_path = input_path[:-4] + '-reconstructed.mp4'
    tensor_to_mp4(output_frames, output_path)
    

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error is commonly used for image reconstruction tasks
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    
if __name__ == '__main__':
    main()


