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


def generate_height_map(image, model):
    height_map = model(image[np.newaxis, ...])[0, ..., 0].numpy()
    return height_map


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

    # Combine color image and normal map using Phong shading
    lit_image = ImageChops.multiply(image, Image.fromarray(normal_map))

    # Adjust brightness to compensate for potential darkening
    enhancer = ImageEnhance.Brightness(lit_image)
    lit_image = enhancer.enhance(2.0)  # Increase brightness by 50%
    return lit_image


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
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建空白画布
    canvas = np.zeros_like(image)

    # 获取图像大小
    height, width, _ = image.shape

    # 模拟油画笔触
    for y in range(0, height, stroke_size):
        for x in range(0, width, stroke_size):
            # 随机选择笔触方向
            angle = np.random.uniform(0, 2 * np.pi)
            dx = int(np.cos(angle) * stroke_length)
            dy = int(np.sin(angle) * stroke_length)

            # 确保笔触在图像范围内
            x1, y1 = x, y
            x2, y2 = min(width - 1, max(0, x + dx)), min(height - 1, max(0, y + dy))

            # 获取颜色
            color = image[y, x].tolist()

            # 画笔触
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

    # 生成笔触效果图像
    brush_stroke_image = apply_brush_strokes(image_path, stroke_size=2, stroke_length=8)


    save_image((brush_stroke_image).astype(np.uint8), os.path.join(result_dir, 'brush_strokes.png'))

    #直接使用生成的笔触图像进行高度图和光照处理
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

    # Save the enhanced image
    save_image((enhanced_image * 255).astype(np.uint8), os.path.join(result_dir, 'improved_lit_image.png'))

    # Optional: Apply style transfer
    apply_style = False  # Set to True to apply style transfer
    if apply_style:
        stylized_image = apply_style_transfer(enhanced_image, style_image_path)
        save_image((stylized_image * 255).astype(np.uint8), os.path.join(result_dir, 'stylized_image2.png'))

    the_current_time()
