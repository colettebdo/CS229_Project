import cv2
import os

image_folder = "data/Kaggle4k/images4k/Dataset4K"
output_video = "4kImageDataset.mov"

output_4k = "data/ImageDataset4k.mov"
output_1080p = "data/ImageDataset1080p.mov"

images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()

first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape

target_width, target_height = 1920, 1080

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30

# Create 4K video
video_4k = cv2.VideoWriter(output_4k, fourcc, fps, (width, height))
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video_4k.write(frame)
video_4k.release()
print("4K dataset video created successfully!")

# Create 1080p video
video_1080p = cv2.VideoWriter(output_1080p, fourcc, fps, (target_width, target_height))
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    frame_resized = cv2.resize(frame, (target_width, target_height))
    video_1080p.write(frame_resized)
video_1080p.release()
print("1080p dataset video created successfully!")

cv2.destroyAllWindows()
