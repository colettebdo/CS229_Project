import torch
import torch.nn as nn
import torch.optim as optim


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

    def forward(self, x):
        '''
        Given a downscaled video x, return an upscaled version of it y. TODO: Revise this into a parallel, non-iterative approach.

        Args:
            x: a tensor of shape=(N_TimeFrames, IN_HEIGHT, IN_WIDTH, N_Pixel_Channels)

        Returns:
            y: a Pytorch tensor of shape=(N_TimeFrames, OUT_HEIGHT, OUT_WIDTH, N_Pixel_Channels) and dtype=uint8
        '''
        T, H, W, C = x.shape
        # Convert video to float32 for processing
        x = x.to(torch.float32)

        # Compute average color for each frame
        avg_colors = torch.mean(x, dim=(1,2))  # shape: (T, C)

        # Initialize output with each frame filled with its average color
        y = avg_colors[:, None, None, :].expand(T, H, 2*W, C)
        y = y.clone()

        # Fill original pixels into expanded positions
        y[:, ::2, ::2] = x[:, ::2]
        y[:, 1::2, 1::2] = x[:, 1::2]

        # Convert back to uint8 format
        y = y.to(torch.uint8)
        return y
        '''
        # INTERATIVE APPROACH
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
        # Convert back to uint8 format
        y = y.to(torch.uint8)
        return y
        '''

