from PIL import Image
import os

def load_images(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))]
    return [Image.open(os.path.join(directory, f)) for f in image_files]

def save_images(image_list):
    for idx, img in enumerate(image_list, start=1):
        # Format the index with zero padding using zfill
        filename = f'23OctoberImage/tank/tank_{str(idx).zfill(6)}.jpg'
        img.save(filename)
        print(f'Image saved as {filename}')

# Load images from a directory
raw_images = load_images('C:/Users/snn23kfl/tank/')  #Replace with the path to your images
save_images(raw_images)
