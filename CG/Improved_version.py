# -*- coding: utf-8 -*-
from PIL import Image, ImageChops, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow_hub as hub
import cv2


def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

# build a simple convolutional neural network model
def build_model():
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = tf.keras.Model(inputs, x)
    return model


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.exif_transpose(image)
    image = np.array(image) / 255.0
    return image

# height map generation
def generate_height_map(image, model):
    height_map = model(image[np.newaxis, ...])[0, ..., 0].numpy()
    return height_map

# apply advanced lighting to the image based on the height map
def apply_lighting(image, height_map):
    image = Image.fromarray((image * 255).astype(np.uint8))
    height_map = Image.fromarray((height_map * 255).astype(np.uint8))

    # Adjust height map size to match the original image
    height_map = height_map.resize(image.size, Image.BILINEAR)

    # Compute surface normals from the height field
    normals = cv2.Sobel(np.array(height_map), cv2.CV_64F, 1, 1, ksize=5)
    normals = (normals - np.min(normals)) / (np.max(normals) - np.min(normals))
    normals = np.clip(normals, 0, 1)
    # Convert normals to an RGB image
    normal_map = cv2.cvtColor((normals * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Apply Parallax Mapping
    parallax_height = 0.1  # Parallax height can be adjusted for more depth
    parallax_normals = normals * parallax_height
    parallax_image = ImageChops.offset(image, int(parallax_normals[..., 0].mean()),
                                       int(parallax_normals[..., 1].mean()))

    # Combine color image and normal map using Phong shading
    lit_image = ImageChops.multiply(parallax_image.convert("RGB"), Image.fromarray(normal_map))

    # Apply Ambient Occlusion
    occlusion = cv2.GaussianBlur(np.array(height_map), (15, 15), 0)
    occlusion = 1 - (occlusion / 255.0)  # Invert and normalize
    occlusion_map = Image.fromarray((occlusion * 255).astype(np.uint8)).convert("L")

    # Ensure occlusion_map is in the same size as lit_image
    occlusion_map = occlusion_map.resize(lit_image.size, Image.BILINEAR)

    # Convert occlusion_map to RGB to match lit_image mode
    occlusion_map = occlusion_map.convert("RGB")

    lit_image = ImageChops.multiply(lit_image, occlusion_map)

    # Adjust brightness to compensate for potential darkening
    enhancer = ImageEnhance.Brightness(lit_image)
    lit_image = enhancer.enhance(3.5)  # Increase brightness by 50%
    return lit_image

# apply laplacian to the image to enhance the details
def enhance_details(image):
    image = np.array(image)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    enhanced_image = cv2.convertScaleAbs(laplacian)
    enhanced_image = cv2.addWeighted(image, 1.0, enhanced_image, 0.5, 0)
    return enhanced_image / 255.0


def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


def apply_brush_strokes(image_path, stroke_size=4, stroke_length=12):
    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # create a blank canvas
    canvas = np.zeros_like(image)

    # get size
    height, width, _ = image.shape

    # simulate brush strokes
    for y in range(0, height, stroke_size):
        for x in range(0, width, stroke_size):
            # randomize stroke length and angle
            angle = np.random.uniform(0, 2 * np.pi)
            dx = int(np.cos(angle) * stroke_length)
            dy = int(np.sin(angle) * stroke_length)

            # make sure the stroke is within the image
            x1, y1 = x, y
            x2, y2 = min(width - 1, max(0, x + dx)), min(height - 1, max(0, y + dy))

            # get color from the image
            color = image[y, x].tolist()

            # draw the stroke
            cv2.line(canvas, (x1, y1), (x2, y2), color, stroke_size)

    return canvas


# Optional style transfer function
def apply_style_transfer(content_image, style_image_path):
    style_image = load_image(style_image_path)
    style_image = tf.convert_to_tensor(style_image, dtype=tf.float32)
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_module(tf.constant(content_image, dtype=tf.float32)[None, ...], style_image[None, ...])[0]
    return stylized_image.numpy().squeeze()


if __name__ == "__main__":
    image_path = './input/Scene.jpg'  # Replace with your image path
    style_image_path = './input/style2.png'  # Replace with your style image path
    result_dir = './output'  # Replace with your desired output directory
    os.makedirs(result_dir, exist_ok=True)

    the_current_time()

    # create the stroke image
    brush_stroke_image = apply_brush_strokes(image_path, stroke_size=2, stroke_length=8)
    save_image((brush_stroke_image).astype(np.uint8), os.path.join(result_dir, 'brush_strokes.png'))

    brush_stroke_image = brush_stroke_image / 255.0
    # Build the simple CNN model
    model = build_model()
    # Generate the height map
    height_map = generate_height_map(brush_stroke_image, model)
    save_image((height_map*255).astype(np.uint8), os.path.join(result_dir, 'height_map.png'))

    # Apply lighting effects
    lit_image = apply_lighting(brush_stroke_image, height_map)

    # Enhance image details
    enhanced_image = enhance_details(lit_image)
    save_image((enhanced_image * 255).astype(np.uint8), os.path.join(result_dir, 'improved_lit_image.png'))

    # Optional: Apply style transfer
    apply_style = False  # Set to True to apply style transfer
    if apply_style:
        stylized_image = apply_style_transfer(enhanced_image, style_image_path)
        save_image((stylized_image * 255).astype(np.uint8), os.path.join(result_dir, 'stylized_image2.png'))

    the_current_time()
