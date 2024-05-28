import os
import cv2
import numpy as np
from gradio_client import Client


def make_black_transparent(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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

    # extract only upper/lower
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


# mask_out_image('images/jeans2.jpg')
mask_out_image('images/tshirt3.jpg', 'upper')



