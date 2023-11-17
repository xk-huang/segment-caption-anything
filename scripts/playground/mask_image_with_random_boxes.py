import random
from PIL import Image, ImageDraw


def mask_image_with_random_boxes(image_path, output_path, num_boxes, box_width, box_height):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Create a drawing object to draw the black boxes
    draw = ImageDraw.Draw(image)

    # Draw the black boxes
    for _ in range(num_boxes):
        x1 = random.randint(0, width - box_width)
        y1 = random.randint(0, height - box_height)
        x2 = x1 + box_width
        y2 = y1 + box_height
        draw.rectangle([x1, y1, x2, y2], fill="black")

    # Save the masked image
    image.save(output_path)


# Example usage:
input_path = "tmp/data/Beautiful-horse-running-in-a-grassy-field.jpg"
output_path = "tmp/data/random_mask-Beautiful-horse-running-in-a-grassy-field.jpg"
mask_image_with_random_boxes(input_path, output_path, 64, 140, 140)
