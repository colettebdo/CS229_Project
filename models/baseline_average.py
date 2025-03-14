import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv


class BaselineAverageModel(nn.Module):
    def __init__(self, in_shape, out_shape):
        '''
        Args:
            in_shape: a tuple of (input_height, input_width)
            out_shape: a tuple of (output_height, output_width)
        '''
        super(BaselineAverageModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input_path, output_path):
        '''
        Given a downscaled video x, return an upscaled version of it y. TODO: Revise this into a parallel, non-iterative approach.

        Args:
            x: a tensor of shape=(N_TimeFrames, IN_HEIGHT, IN_WIDTH, N_Pixel_Channels)

        Returns:
            y: a Pytorch tensor of shape=(N_TimeFrames, OUT_HEIGHT, OUT_WIDTH, N_Pixel_Channels) and dtype=uint8
        '''
        cap = cv.VideoCapture(input_path)
    
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        channels = 3  # Assume RGB
        upscale_width = width * 2  # Expanding width
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (upscale_width, height), isColor=True)
        f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video.")
                break
            
            x = torch.tensor(frame, dtype=torch.float32)
            x = x.unsqueeze(0)
            _, H, W, C = x.shape

            avg_colors = torch.mean(x, dim=(1, 2))
            
            # Create an upscaled frame filled with the average color
            y = avg_colors[:, None, None, :].expand(1, H, 2 * W, C).clone()

            # Even frames - Checkerboard pattern
            if f % 2 == 0:
                y[:, ::2, 1::2] = x[:, ::2]  # Even rows
                y[:, 1::2, ::2] = x[:, 1::2]  # Odd rows
            else:
                # Odd frames - Inverted checkerboard pattern
                y[:, ::2, ::2] = x[:, ::2]  # Even rows
                y[:, 1::2, 1::2] = x[:, 1::2]  # Odd rows

            y = y.squeeze(0).to(torch.uint8).numpy()  # Remove batch dim → (H, 2W, C)

            out.write(y)
            f += 1

        cap.release()
        out.release()
        cv.destroyAllWindows()
