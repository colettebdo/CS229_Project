# CS229_Project

## Milestone (2/7)

* Goals
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

