import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

sys.path.append('video-mae')
import models_mae


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def reconstruct_frame(img, model):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    im_paste = x * (1 - mask) + y * mask
    return torch.clip((im_paste * imagenet_std + imagenet_mean) * 255, 0, 255).int()


def get_masked(img, model):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    _, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    im_masked = x * (1 - mask)

    return torch.clip((im_masked * imagenet_std + imagenet_mean) * 255, 0, 255).int()


def process_video_mae(input_path, output_path):
        model_mae_gan = prepare_model("video-mae\project\mae_visualize_vit_large_ganloss.pth", 'mae_vit_large_patch16')

        cap = cv.VideoCapture(input_path)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        print(width, height)
        i = 0
        while cap.isOpened() and i < 220:
            if i < 200: 
                i += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break

            new_img = np.zeros((2016, 3808, 3), dtype=np.uint8)

            for y in range(9):
                for x in range(17):
                    img = np.array(frame)[y*224:(y+1)*224, x*224:(x+1)*224] / 255.

                    img = img - imagenet_mean
                    img = img / imagenet_std
                    
                    new_img[y*224:(y+1)*224, x*224:(x+1)*224] = reconstruct_frame(img, model_mae_gan).int()[0]

            reconstructed_frame = cv.resize(new_img, (width, height))
            out.write(reconstructed_frame)
            i+=1

        cap.release()
        out.release()
        print(f"Reconstructed video saved to {output_path}")


def process_image_mae(input_image_path):
    input_image = cv.imread(input_image_path)
    if input_image is None:
        raise ValueError("Failed to load image. Ensure the file is a valid image.")


    model_mae_gan = prepare_model("video-mae/project/mae_visualize_vit_large_ganloss.pth", 'mae_vit_large_patch16')
    new_img = np.zeros((2016, 3808, 3), dtype=np.uint8)

    for y in range(9):
        for x in range(17):
            img_patch = np.array(input_image)[y*224:(y+1)*224, x*224:(x+1)*224] / 255.
            img_patch = img_patch - imagenet_mean
            img_patch = img_patch / imagenet_std

            new_img[y*224:(y+1)*224, x*224:(x+1)*224] = reconstruct_frame(img_patch, model_mae_gan).int()[0]

    reconstructed_image = cv.resize(new_img, (input_image.shape[1], input_image.shape[0]))

    directory, filename = os.path.split(input_image_path)
    name, ext = os.path.splitext(filename)
    output_image_path = os.path.join(directory, f"{name}_reconstructued_mae{ext}")
    cv.imwrite(output_image_path, reconstructed_image)

    print(f"Reconstructed image saved to {output_image_path}")
    return output_image_path
