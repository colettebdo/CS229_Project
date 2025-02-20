import numpy as np
import cv2 as cv

HEIGHT = 2160
WIDTH = 3840

def checkerboard(height, width):
    checkboard = np.zeros((height, width), dtype=bool)
    checkboard[1::2, ::2] = True
    checkboard[::2, 1::2] = True
    return checkboard

def custom_downsample(path, mask_func, frame_count):
    cap = cv.VideoCapture(path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{path[:-4]}-checkerboard.mov', fourcc, 24.0, (WIDTH // 2, HEIGHT), isColor=True)

    mask = mask_func(HEIGHT, WIDTH)
    mask_inv = np.invert(mask)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or i > frame_count:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to numpy array if needed
        frame = np.array(frame, dtype=np.uint8)

        # Create a downsampled frame
        checkerboard_frame = np.zeros((HEIGHT, WIDTH // 2, 3), dtype=np.uint8)

        if cap.get(cv.CAP_PROP_POS_FRAMES) % 2 == 0:
            checkerboard_frame[:, :] = frame[mask].reshape(HEIGHT, WIDTH // 2, 3)
        else:
            checkerboard_frame[:, :] = frame[mask_inv].reshape(HEIGHT, WIDTH // 2, 3)

        out.write(checkerboard_frame)
        i += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()

def rerender(path, frame_count):
    cap = cv.VideoCapture(path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{path[:-4]}-native.mov', fourcc, 24.0, (WIDTH, HEIGHT), isColor=True)
    i = 0

    while cap.isOpened() and i < frame_count:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        out.write(frame)

        # cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        i += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()

def downsample(input_path, frame_count, target_width=1920, target_height=1080):
    """
    Downscales a 4K video to 1080p resolution.

    Args:
        input_path (str): Path to the input 4K video.
        output_path (str): Path to save the 1080p video.
        target_width (int): Desired width of the output video (default: 1920).
        target_height (int): Desired height of the output video (default: 1080).
    """
    # Open the video file
    cap = cv.VideoCapture(input_path)
    
    # Get original frame rate
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
    out = cv.VideoWriter(f'{input_path[:-4]}-1080p.mov', fourcc, fps, (target_width, target_height), isColor=True)

    # Process frames one by one
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or i > frame_count:
            print("Finished processing video.")
            break

        # Resize frame to 1080p
        frame_resized = cv.resize(frame, (target_width, target_height), interpolation=cv.INTER_AREA)

        # Write the resized frame
        out.write(frame_resized)
        i += 1

    # Release resources
    cap.release()
    out.release()
    cv.destroyAllWindows()

def main():
    video_path = "data\LifeUntouched_P3_4K_PQ_XQ.mov"
    custom_downsample(video_path, checkerboard, 500)
    rerender(video_path, 500)
    # downsample(video_path, 500)

if __name__ == '__main__':
    main()
