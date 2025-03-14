import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2 as cv


class ConvolvedAverageModel(nn.Module):
    def __init__(self, in_shape, out_shape):
        '''
        Args:
            in_shape: a tuple of (input_height, input_width)
            out_shape: a tuple of (output_height, output_width)
        '''
        super(ConvolvedAverageModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

    def upsample_video(self, x, f):
        """
        Upscales a checkerboard-subsampled video frame by filling in missing pixels
        using an average filter (mean of top, bottom, left, right).

        Args:
            x (torch.Tensor): Input frame of shape (1, H, W_half, C) in float32.

        Returns:
            torch.Tensor: Upscaled frame of shape (1, H, 2*W_half, C) in uint8.
        """
        _, H, W_half, C = x.shape
        W = W_half * 2
        y = torch.zeros((1, H, W, C), dtype=x.dtype, device=x.device)

        # Even frames - Checkerboard pattern
        if f % 2 == 0:
            y[:, ::2, 1::2] = x[:, ::2]  # Even rows
            y[:, 1::2, ::2] = x[:, 1::2]  # Odd rows
        else:
            # Odd frames - Inverted checkerboard pattern
            y[:, ::2, ::2] = x[:, ::2]  # Even rows
            y[:, 1::2, 1::2] = x[:, 1::2]  # Odd rows

        y_p = y.permute(0, 3, 1, 2)

        # Define the mean filter kernel
        kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        
        kernel = kernel.repeat(C, 1, 1, 1)  # Repeat for each channel (R, G, B)
        
        conv_output = F.conv2d(y_p, kernel, padding=1, groups=C) / 4  # Normalize sum to get mean
        
        y_p = torch.where(y_p == 0, conv_output, y_p)
        y = y_p.permute(0, 2, 3, 1).to(torch.uint8)

        return y

    def forward(self, path, output_path):
        '''
        Given a downscaled video x, return an upscaled version of it y. TODO: Revise this into a parallel, non-iterative approach.

        Args:
            x: a tensor of shape=(N_TimeFrames, IN_HEIGHT, IN_WIDTH, N_Pixel_Channels)

        Returns:
            y: a Pytorch tensor of shape=(N_TimeFrames, OUT_HEIGHT, OUT_WIDTH, N_Pixel_Channels) and dtype=uint8
        '''
        cap = cv.VideoCapture(path)

        fps = int(cap.get(cv.CAP_PROP_FPS))
        H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        W_half = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        W = W_half * 2

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(f'{output_path}', fourcc, fps, (W, H), isColor=True)
        
        i = 0
        while cap.isOpened() and i < 500:
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = torch.from_numpy(frame).to(torch.float32).unsqueeze(0)  # (1, H, W_half, C)

            upscaled_frame = self.upsample_video(frame, i)  # Calls the previous upsampling function
            upscaled_frame_np = upscaled_frame.squeeze(0).cpu().numpy()
            out.write(upscaled_frame_np)

            # cv.imshow('frame', upscaled_frame_np)  # Optional for debugging
            if cv.waitKey(1) == ord('q'):
                break
            i += 1

        cap.release()
        out.release()
        cv.destroyAllWindows()
        print(f"Upscaled video saved to {output_path}")