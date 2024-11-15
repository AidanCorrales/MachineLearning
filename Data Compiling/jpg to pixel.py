from PIL import Image


def save_pixel_values(image_path, output_file):
    # Open the image using Pillow
    img = Image.open(image_path)

    # Convert the image to grayscale (if it's not already)
    img = img.convert('L')  # 'L' mode is for grayscale

    # Resize the image to ensure it's 48x48
    img = img.resize((48, 48))

    # Get pixel values
    pixels = list(img.getdata())

    # Open the output file to write pixel values
    with open(output_file, 'w') as f:
        for pixel in pixels:
            # Convert pixel value to either 0 (black) or 255 (white)
            pixel_value = pixel
            f.write(f"{pixel_value} ")

    print(f"Pixel values saved to {output_file}")


# Example usage
image_path = 'input_image.jpg'  # Replace with your image path
output_file = 'pixel_values.txt'  # The file where the pixel values will be saved
save_pixel_values(image_path, output_file)
