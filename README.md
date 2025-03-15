# CS229_Project

This repository is for a CS229 Final Project. The models we trained can be found in the `models` directory. The `train.ipynb` was used for all training. The files in the `video-mae` directory show our modified version of Facebook's MAE (https://github.com/facebookresearch/mae). The lightweight SRGAN model was inspired by Ledig et al. (https://arxiv.org/abs/1609.04802v5). The FSRCNN model was inspired by Dong et al. (https://arxiv.org/pdf/1608.00367v1) but also modified for our specific task. All other files were developed by us for our project.

## Timeline and Brainstorming

### Milestone (2/21)

* Tasks
    1. Import videos
    2. Code to preprocess data (downsampling)
        * "black-out" frames using checkerboard approach
    3. Implement Normal Average Baseline
        * Input: downsampled video
        * Output: upscaled video
        * Method: Replace each black-out pixel with frame pixel average 
    4. Implement Local Average Baseline
        * Input: downsampled video
        * Output: upscaled video
        * Method: Replace each black-out pixel with local region's pixel average
    5. Code to evaluate model
        * Similarity: use MSE between original video with upscaled video output
        * Compression: ratio for (downsampled video + model) / (original video size)
    6. Implement CNN Model
        * just upscale model?
        * downscale model too?

### Final Report (3/14)

* Tasks
    1. FSRCNN
    2. SRGAN
    3. MAE (ViT)

### Poster Session (3/19)
