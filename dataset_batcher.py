import cv2
import os

input_video_4k =  "data/ImageDataset4k.mov"
input_video_1080p = "data/ImageDataset1080p.mov"
frame_chunk_size = 100
output_folder = "data"

os.makedirs(output_folder, exist_ok=True)

def split_video(input_video):
    video_capture = cv2.VideoCapture(input_video)
    if not video_capture.isOpened():
        print(f"Error opening video file {input_video}")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: FPS value not found for {input_video}")
        return

    chunk_number = 1
    
    while True:
        chunk_frames = []
        for _ in range(frame_chunk_size):
            ret, frame = video_capture.read()
            if not ret:
                break
            chunk_frames.append(frame)
        
        if not chunk_frames:
            break

        output_video_chunk = os.path.join(output_folder, f"{os.path.basename(input_video).split('.')[0]}_chunk_{chunk_number}.mov")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for better compatibility
        height, width, _ = chunk_frames[0].shape
        video_writer = cv2.VideoWriter(output_video_chunk, fourcc, fps, (width, height))

        for frame in chunk_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"{os.path.basename(input_video)} chunk {chunk_number} created successfully!")
        chunk_number += 1

    video_capture.release()

split_video(input_video_4k)
split_video(input_video_1080p)

cv2.destroyAllWindows()
print("Dataset batching completed.")
