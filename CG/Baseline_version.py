import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.exif_transpose(image)
    return np.array(image) / 255.0


# simulate painting with line brush strokes
def apply_brush_strokes(image, stroke_size=4, stroke_length=12):
    # start with a blank canvas
    canvas = np.zeros_like(image)
    height, width, _ = image.shape

   # simulate oil painting brush strokes
    for y in range(0, height, stroke_size):
        for x in range(0, width, stroke_size):
            angle = np.random.uniform(0, 2 * np.pi)
            dx = int(np.cos(angle) * stroke_length)
            dy = int(np.sin(angle) * stroke_length)
            x1, y1 = x, y
            x2, y2 = min(width - 1, max(0, x + dx)), min(height - 1, max(0, y + dy))
            color = image[y, x].tolist()
            cv2.line(canvas, (x1, y1), (x2, y2), color, stroke_size)
    return canvas

# simulate a 3D surface where the brightness of each pixel represents the height at that point
def generate_height_map(image):
    height_map = np.zeros_like(image[:, :, 0], dtype=np.float32)
    height_increment = 1.0 / (np.max(image) + 1)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            height_map[y, x] = height_map[y, x] + height_increment * np.mean(image[y, x])
    return height_map

# compute the normals of surface based on the height map
def compute_normals(height_map):
    dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=5)
    normal_map = np.dstack((-dx, -dy, np.ones_like(height_map)))
    normal_map = cv2.normalize(normal_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normal_map

# apply lighting to the image based on the normal map
def apply_lighting(image, normal_map):
    normal_map_rgb = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    normal_map_rgb = normal_map_rgb / 255.0

    # convert image and normal map to float32 for better precision
    image_float = image.astype(np.float32)
    normal_map_rgb_float = normal_map_rgb.astype(np.float32)

    lit_image = cv2.addWeighted(image_float, 0.5, normal_map_rgb_float, 0.5, 0)
    lit_image = np.clip(lit_image, 0, 1)  # clip values to the range [0, 1]

    lit_image = Image.fromarray((lit_image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(lit_image)

    return lit_image


def save_image(image, path):
    image.save(path)


if __name__ == "__main__":
    image_path = './input/Scene.png'  # Replace with your image path

    result_dir = './output'
    os.makedirs(result_dir, exist_ok=True)

    # load image and apply brush strokes
    image = load_image(image_path)
    brush_stroke_image = apply_brush_strokes(image, stroke_size=2, stroke_length=8)

    # generate and save the height map
    height_map = generate_height_map(brush_stroke_image)
    save_image(Image.fromarray((height_map * 255).astype(np.uint8)),
               os.path.join(result_dir, 'baseline_height_map.png'))

    # compute the normal of surface using the height map
    normal_map = compute_normals(height_map)

    # calculate and apply the illumination to the image
    lit_image = apply_lighting(brush_stroke_image, normal_map)

    # save images
    # save_image(Image.fromarray(normal_map), os.path.join(result_dir, 'baseline_normal_map_protrait.png'))
    save_image(lit_image, os.path.join(result_dir, 'baseline_lit_image.png'))
