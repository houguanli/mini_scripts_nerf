import os
import numpy as np
from PIL import Image


def crop_images_in_folder(input_folder, output_folder):
    """Crops all PNG images in the input folder and saves them to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Load image
                image = Image.open(image_path).convert("RGBA")
                image_data = np.array(image)

                # Identify non-transparent pixels
                alpha_channel = image_data[:, :, 3]  # Alpha channel
                non_transparent_mask = alpha_channel > 0

                if not np.any(non_transparent_mask):
                    print(f"Skipping {filename}: No visible content found.")
                    continue

                # Find the bounding box of non-transparent content
                rows = np.any(non_transparent_mask, axis=1)
                cols = np.any(non_transparent_mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                # Crop and save the image
                cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
                cropped_image.save(output_path)
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Processing complete.")

# Example usage:
input_path  = "C:/Users/guanl/Desktop/reg_paper_writing/supplement_matirials/all_results/ref_image/wo_bk"
crop_images_in_folder(input_path, input_path)