from rembg import remove
from PIL import Image
import os

input_folder = 'uploads'
output_folder = 'transparent'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_transparent.png')

        print(f"Processing {input_path} ...")

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGBA")
                result = remove(img)
                result.save(output_path)
                print(f"✅ Saved transparent image to {output_path}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
