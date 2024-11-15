import numpy as np
from PIL import Image


def load_pixels_from_txt(input_file):
    # Read the pixel values from the text file
    with open(input_file, 'r') as f:
        pixels = f.read().split()

    # Convert the list of strings to integers
    pixels = list(map(int, pixels))

    # Ensure that we have exactly 2304 pixels (48x48 image)
    if len(pixels) != 2304:
        raise ValueError(
            "The input file does not contain the correct number of pixels (2304 expected for a 48x48 image).")

    # Reshape the pixel list into a 48x48 image (48 rows and 48 columns)
    pixels = np.array(pixels).reshape((48, 48))

    return pixels


def save_image_from_pixels(pixels, output_image):
    # Create a Pillow image from the pixel values
    img = Image.fromarray(pixels.astype(np.uint8))  # Ensure the pixel data is in the correct type (uint8)

    # Save the image as a JPG file
    img.save(output_image, 'JPEG')
    print(f"Image saved to {output_image}")


# Example usage
input_file = 'pixel_values.txt'  # The txt file containing pixel values
output_image = 'output_image.jpg'  # The file to save the reconstructed image
pixels = load_pixels_from_txt(input_file)
save_image_from_pixels(pixels, output_image)
