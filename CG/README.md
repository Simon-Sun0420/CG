# Group members :
Boya Wang(u7632096), Jiahong Sun(u7551207), Diming Xu(u7705332)




# Enhanced Oil Painting Brushstroke Rendering in Images

This project focuses on enhancing the oil painting brushstroke effect in images using advanced image processing techniques. We provide both a baseline implementation and an improved implementation that incorporates style transfer to achieve more realistic and visually appealing results.

Our official repo locates at: https://github.com/Simon-Sun0420/CG

# Input folder

The input folder contains all the raw images that are the starting point of the image rendering process. These raw images are used as input data for our rendering algorithms to analyze and process in order to generate a rendered image with the effect of oil paint strokes. The user can find all the base images used to generate the effect in this folder.

# Output folder

The output folder holds the images that are processed by our rendering system. These images show the final result of the algorithmic processing, including the application of height map generation, parallax mapping, ambient occlusion techniques and Laplace filter enhancement. Due to space limitations, some of the images in this folder were not included in the final paper. Users can view all renderings in this folder, including those additional images that could not be shown in the paper due to space limitations.


## Project Structure

- `Baseline_version.py`: Contains the baseline implementation of the oil painting brushstroke rendering.
```bash
python Baseline_version.py
```

- `Improved_version.py`: Contains the improved implementation with additional features, including style transfer.
    You could set the apply style transfer flag in main( ) to True to enable the style transfer feature.
```bash
python Improved_version.py
```


## Requirements

To run the project, you need to set up a python env of 3.9.

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

