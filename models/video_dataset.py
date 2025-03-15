import torch
import cv2 as cv
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, checkerboard_path, full_video_path, max_frames=None, transform=None, stride=10):
        self.checkerboard_path = checkerboard_path
        self.full_video_path = full_video_path
        self.transform = transform
        self.max_frames = max_frames
        self.stride = stride  # Process every `stride` frames per epoch
        self.epoch_offset = 0

        self.cap_cb = cv.VideoCapture(checkerboard_path)
        self.cap_full = cv.VideoCapture(full_video_path)

        self.total_frames = int(min(self.cap_cb.get(cv.CAP_PROP_FRAME_COUNT), 
                                    self.cap_full.get(cv.CAP_PROP_FRAME_COUNT)))

        if self.max_frames:
            self.total_frames = min(self.total_frames, self.max_frames)

    def set_epoch(self, epoch):
        self.epoch_offset = epoch % self.stride  # Cycle through frame offsets

    def __len__(self):
        return (self.total_frames - self.epoch_offset) // self.stride  # Frames per epoch

    def __getitem__(self, idx):
        frame_idx = self.epoch_offset + idx * self.stride
        # print(f"frame: {frame_idx}")
        if frame_idx >= self.total_frames:
            raise IndexError("Frame index out of range.")

        self.cap_cb.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        self.cap_full.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

        ret_cb, frame_cb = self.cap_cb.read()
        ret_full, frame_full = self.cap_full.read()

        if not ret_cb or not ret_full:
            raise RuntimeError(f"Error reading frame {frame_idx}")

        frame_cb = torch.from_numpy(frame_cb).float().permute(2, 0, 1) / 255.0
        frame_full = torch.from_numpy(frame_full).float().permute(2, 0, 1) / 255.0

        if self.transform:
            frame_cb = self.transform(frame_cb)
            frame_full = self.transform(frame_full)

        return frame_cb, frame_full
    
    def close(self):
        self.cap_cb.release()
        self.cap_full.release()
