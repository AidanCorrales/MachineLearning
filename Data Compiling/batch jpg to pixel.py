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
    with open(output_file, 'a') as f:  # 'a' for append mode
        for pixel in pixels:
            # Write the pixel value to the file
            f.write(f"{pixel} ")
        f.write(",\n")

    print(f"Pixel values saved to {output_file}")

def process_multiple_images(num_images, output_file):
    # Iterate through image files, starting at '001.jpeg' to 'num_images.jpeg'
    for i in range(1, num_images + 1):
        # Format the filename with leading zeros (e.g., 001.jpeg, 002.jpeg, ...)
        image_filename = f"angry/{i:03}.jpg"
        save_pixel_values(image_filename, output_file)

# Example usage
num_images = 100  # Modify to the number of images you want to process
output_file = 'angry.txt'  # The file where the pixel values will be saved
process_multiple_images(num_images, output_file)
