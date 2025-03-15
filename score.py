import cv2 as cv
import numpy as np
import torch

def calculate_mse(original_path, reconstructed_path):
    """
    Computes the Mean Squared Error (MSE) between an original video and its reconstructed version.

    Args:
        original_path (str): Path to the original high-resolution video.
        reconstructed_path (str): Path to the reconstructed video.

    Returns:
        float: The average MSE over all frames.
    """
    cap_orig = cv.VideoCapture(original_path)
    cap_recon = cv.VideoCapture(reconstructed_path)
    if not cap_orig.isOpened() or not cap_recon.isOpened():
        raise ValueError("Error opening one or both video files.")

    total_mse = 0
    frame_count = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_recon, frame_recon = cap_recon.read()

        if not ret_orig or not ret_recon:
            break

        if frame_orig.shape != frame_recon.shape:
            raise ValueError(f"Frame size mismatch: {frame_orig.shape} vs {frame_recon.shape}")

        mse = np.sum((frame_orig.astype(np.float32) - frame_recon.astype(np.float32)) ** 2)
        total_mse += mse
        frame_count += 1

    cap_orig.release()
    cap_recon.release()
    cv.destroyAllWindows()

    if frame_count == 0:
        raise ValueError("No frames found in one or both videos.")
    
    return total_mse / frame_count
    