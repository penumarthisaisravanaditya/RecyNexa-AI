import os
import shutil

SOURCE = "dataset/garbage_classification"
DESTINATION = "dataset"

# Mapping from downloaded dataset → RecyNexa classes
class_mapping = {
    "battery": "e_waste",
    "biological": "organic",
    "cardboard": "cardboard",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
}

# Create destination folders
for target_class in set(class_mapping.values()):
    os.makedirs(os.path.join(DESTINATION, target_class), exist_ok=True)

total_copied = 0

for source_class, target_class in class_mapping.items():

    source_folder = os.path.join(SOURCE, source_class)
    target_folder = os.path.join(DESTINATION, target_class)

    if not os.path.exists(source_folder):
        print(f"WARNING: {source_folder} not found")
        continue

    files = os.listdir(source_folder)

    copied = 0

    for filename in files:

        source_file = os.path.join(source_folder, filename)

        if os.path.isfile(source_file):
            destination_file = os.path.join(
                target_folder,
                f"{source_class}_{filename}"
            )

            shutil.copy2(source_file, destination_file)
            copied += 1
            total_copied += 1

    print(f"{source_class:15} -> {target_class:12} : {copied} images")

print("\nDataset preparation completed!")
print(f"Total images copied: {total_copied}")