import numpy as np
import cv2 as cv

HEIGHT = 2160
WIDTH = 3840

def checkerboard(height, width):
    checkboard = np.zeros((height, width), dtype=bool)
    checkboard[1::2, ::2] = True
    checkboard[::2, 1::2] = True
    return checkboard

def downsample_video(path, mask_func):
    cap = cv.VideoCapture(path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{path[:-4]}-small.mov', fourcc, 24.0, (WIDTH // 2, HEIGHT), isColor=True)
    i = 0

    mask = mask_func(HEIGHT, WIDTH)

    while cap.isOpened() and i < 1000:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        checkerboard_frame = frame[mask]
        
        checkerboard_frame_resized = checkerboard_frame.reshape((HEIGHT, WIDTH // 2, 3))

        out.write(checkerboard_frame_resized)

        # cv.imshow('frame', checkerboard_frame_resized)
        if cv.waitKey(1) == ord('q'):
            break
        i += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()

def rerender_video(path):
    cap = cv.VideoCapture(path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{path[:-4]}-native.mov', fourcc, 24.0, (WIDTH, HEIGHT), isColor=True)
    i = 0

    while cap.isOpened() and i < 1000:
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

def main():
    downsample_video("data/Seconds_That_Count.mov", checkerboard)
    rerender_video("data/Seconds_That_Count.mov")

if __name__ == '__main__':
    main()
