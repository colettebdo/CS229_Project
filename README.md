# CS229_Project

This repository is for a CS229 Final Project. The models we trained can be found in the `models` directory. The `train.ipynb` was used for all training. The files in the `video-mae` directory shows our modified version of Facebook's MAE (https://github.com/facebookresearch/mae), we update specifically the masking algorithm and added an implementation for large images. The lightweight SRGAN model was inspired by Ledig et al. (https://arxiv.org/abs/1609.04802v5) but greatly downsized for training. The FSRCNN model was inspired by Dong et al. (https://arxiv.org/pdf/1608.00367v1) but also modified for our specific task. All other files were developed by us for our project.

## Guide:

* Major Files:
    1. cnn.py (Contains our modified implementation of FSRCNN)
    2. srgan.py (Contrains our downsized implementation of SRGAN)
    3. tiled_mae.py + grid_sampling.py (Contains our large image augment of Facebook's pretrain MAE and modified masking algorithm)
    4. convolved_average.py + baseline_average.py (Contains our baselines)

* Other files:
    1. train.ipynb (Notebook where all local training and inference was conducted)
    2. video_dataset.py (Utils for loading video frames from dataset)
    3. preprocess.py (Utils for modifying or loading video frames prior to model)
    4. score.py (Utils for MSE loss functions and image comparison)
    5. dataset_builder.py (Util for taking images and constructing a video file for training)
    6. dataset_batcher.py (Util for splitting video dataset into managable pieces for GPU memory)
    7. reconstruct.py (Deprecated--Utils for reconstructing on baseline)
    8. andrew_tests.py (Deprecated--Used for testing SRGAN before final implementation)

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
