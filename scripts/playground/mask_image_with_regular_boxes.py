from PIL import Image, ImageDraw


def mask_image_with_custom_boxes(image_path, output_path, rows, columns, start_x, start_y, box_width, box_height):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Create a drawing object to draw the black boxes
    draw = ImageDraw.Draw(image)

    # Calculate the x and y step sizes
    step_x = (width - start_x * 2 - box_width) // (columns - 1)
    step_y = (height - start_y * 2 - box_height) // (rows - 1)

    # Draw the black boxes
    for i in range(columns):
        for j in range(rows):
            x1 = start_x + i * step_x
            y1 = start_y + j * step_y
            x2 = x1 + box_width
            y2 = y1 + box_height
            draw.rectangle([x1, y1, x2, y2], fill="black")

    # Save the masked image
    image.save(output_path)


# Example usage:
intput_path = "tmp/data/Beautiful-horse-running-in-a-grassy-field.jpg"
output_path = "tmp/data/mask-Beautiful-horse-running-in-a-grassy-field.jpg"
mask_image_with_custom_boxes(intput_path, output_path, 4, 7, 50, 50, 140, 140)
