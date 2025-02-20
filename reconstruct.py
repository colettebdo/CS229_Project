# This program takes a downsampled video and constructs an upscaled version using the specified model.
import cv2 as cv
import numpy as np
import torch

from models.baseline_average import BaselineAverageModel

IN_HEIGHT = 2160
IN_WIDTH = 1920
OUT_HEIGHT = 2160
OUT_WIDTH = 3840


def mp4_to_nparray(video_path):
    '''
    Converts an mp4 into a np array. 
    Args:
        video_path [string]: path to the mp4 video that we want to convert
    Returns:
        video: a np array of shape=(N_TimeFrames, Height, Width, N_Pixel_Channel)
    '''
    cap = cv.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frames.append(frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

    return np.array(frames)


def nparray_to_mp4(video_array, save_path, codec='mp4v', fps=24):
    '''
    Converts a np array into an mp4 file. 
    Args:
        video_array: a np array of shape (N_TimeFrames, Height, Width, N_Pixel_Channel)
        save_path [string]: path to save the mp4 video that we create
        fps [int]: frames per second for the video
    Returns:
        None
    '''
    T, H, W, C = video_array.shape
    fourcc = cv.VideoWriter_fourcc(*codec)
    out = cv.VideoWriter(save_path, fourcc, fps, (W, H), isColor=True)
    for t in range(T):
        out.write(video_array[t])
    out.release()


def main():
    # Load input video, ensure sizing is correct
    input_path = "data/Seconds_That_Count-small.mov"
    input_frames = mp4_to_nparray(input_path)
    input_frames = input_frames[25:75]  # TODO: ONLY DO 0.5 SECONDS FOR NOW; MY MAC CAN'T PROCESS LARGER
    if (input_frames.shape[1:3] != (IN_HEIGHT, IN_WIDTH)):
        raise ValueError(f'Input video of frame size {input_frames.shape[1:3]} does not match the expected size of ({IN_HEIGHT}, {IN_WIDTH}).')
    print(f"Loading complete: {input_path} is loaded.")

    # Instantiate model
    model = BaselineAverageModel(in_shape=(IN_HEIGHT, IN_WIDTH), out_shape=(OUT_HEIGHT, OUT_WIDTH))

    # Upscale video, ensure sizing is correct
    output_frames = model(torch.Tensor(input_frames))
    output_frames = output_frames.cpu().numpy()
    if (output_frames.shape[1:3] != (OUT_HEIGHT, OUT_WIDTH)):
        raise ValueError(f'Output video of frame size {output_frames.shape[1:3]} does not match the expected size of ({OUT_HEIGHT}, {OUT_WIDTH}).')

    # Save output video
    output_path = input_path[:-4] + '-reconstructed.mp4'
    nparray_to_mp4(output_frames, output_path)
    print(f"Reconstruction complete (Baseline): {output_path}")


if __name__ == '__main__':
    main()