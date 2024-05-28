import os
import cv2
import numpy as np
from gradio_client import Client
from PIL import Image, ImageFilter

def add_border(image_path, border_size):
    # Open the image
    img = Image.open(image_path)

    # Make sure the image has an alpha channel (transparency)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Create a mask using the alpha channel
    mask = img.split()[3].point(lambda x: 255 if x > 0 else 0)

    # Expand the mask to include the border
    expanded_mask = mask.filter(ImageFilter.MaxFilter(border_size))

    # Create a solid white border
    border = Image.new('RGBA', img.size, (255, 255, 255, 0))
    border.paste(img, (0, 0), mask=expanded_mask)

    # Crop the border to the original image size
    border = border.crop((border_size, border_size, border.size[0] - border_size, border.size[1] - border_size))

    return border



def make_black_transparent(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image has an alpha channel
    if image.shape[2] == 3:
        # Add an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Create a mask where black pixels are detected
    black = np.all(image[:, :, :3] == [0, 0, 0], axis=-1)
    
    # Set alpha channel to 0 where the mask is true (black pixels)
    image[black, 3] = 0
    
    # Save the resulting image
    cv2.imwrite(output_path, image)


def mask_out_image(image_path, part_to_extract=None):

    base_name = os.path.basename(image_path)
    image_base_name, _ = os.path.splitext(base_name)

    # get mask from huggingface model
    client = Client("https://wildoctopus-cloth-segmentation.hf.space/")
    result = client.predict(image_path, fn_index=1)
    print(result)
    mask_image_path = result
    # mask_image_path = 'images/dress2_model_mask.png'

    # apply mask image over original image
    src1 = cv2.imread(image_path)
    src2 = cv2.imread(mask_image_path)
    src2 = cv2.resize(src2, src1.shape[1::-1])
    cv2.imwrite(f'images/{image_base_name}_model_mask.png', src2)

    # extract only shirt
    if part_to_extract == "upper":
        target_color = np.array([0, 0, 128]) # Replace B, G, R with the actual values
        tolerance = 0
        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        binary_mask = cv2.inRange(src2, lower_bound, upper_bound)
        cv2.imwrite(f'images/{image_base_name}_model_mask_upper.png', binary_mask)
    elif part_to_extract == "lower":
        target_color = np.array([0, 128, 0]) # Replace B, G, R with the actual values
        tolerance = 0
        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        binary_mask = cv2.inRange(src2, lower_bound, upper_bound)
        cv2.imwrite(f'images/{image_base_name}_model_mask_lower.png', binary_mask)
    else:
        gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'images/{image_base_name}_model_mask_full.png', binary_mask)

    dst = cv2.bitwise_and(src1, src1, mask=binary_mask)
    cv2.imwrite(f'images/{image_base_name}_extracted.png', dst)

    make_black_transparent(f'images/{image_base_name}_extracted.png', f'images/{image_base_name}_extracted_transparent.png')

    bordered_image = add_border(f'images/{image_base_name}_extracted_transparent.png', 19)
    bordered_image.save(f'images/{image_base_name}_bordered.png')

# mask_out_image('images/jeans2.jpg')
mask_out_image('images/tshirt3.jpg', 'upper')





# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# if len(contours) == 0:
#     print("No contours found in the mask image.")

# x, y, w, h = cv2.boundingRect(contours[0])

# cropped_image = dst[y:y+h, x:x+w]
# cv2.imwrite('images/extracted2.png', cropped_image)



