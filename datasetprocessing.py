import os
import cv2
import numpy as np

# Paths to your dataset directories
train_dir = r"C:\Users\urvis\OneDrive\Desktop\AIML_EL\backup\DATASET\TRAIN"
test_dir = r"C:\Users\urvis\OneDrive\Desktop\AIML_EL\backup\DATASET\TEST"

# Function to check if an image is a cartoon
# (This is a placeholder function, as we don't have a cartoon detection model)
def is_cartoon(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        avg_color = np.mean(img)
        # Simple heuristic: very bright or very colorful images might be cartoons
        return avg_color > 200  # Adjust threshold as needed

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

# Function to remove cartoon images from a directory
def remove_cartoons(directory):
    cartoon_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                if is_cartoon(image_path):
                    os.remove(image_path)
                    cartoon_count += 1
                    print(f"Removed cartoon image: {image_path}")
    return cartoon_count

# Remove cartoon images from train and test directories
print("Removing cartoon images from TRAIN directory...")
train_cartoon_count = remove_cartoons(train_dir)

print("Removing cartoon images from TEST directory...")
test_cartoon_count = remove_cartoons(test_dir)

print(f"Total cartoon images removed from TRAIN: {train_cartoon_count}")
print(f"Total cartoon images removed from TEST: {test_cartoon_count}")
