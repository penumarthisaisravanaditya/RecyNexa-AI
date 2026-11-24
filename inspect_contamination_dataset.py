
from pathlib import Path
import zipfile
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "contamination_dataset"

CSV_PATH = DATA_DIR / "dataset.csv"
SPLITS_PATH = DATA_DIR / "object_splits.csv"
ZIP_PATH = DATA_DIR / "color_img.zip"

# Load annotations
df = pd.read_csv(CSV_PATH)
splits = pd.read_csv(SPLITS_PATH)

print("\n=== CSV INFORMATION ===")
print("Dataset rows:", len(df))
print("Dataset columns:", df.columns.tolist())
print("\nContaminates values:")
print(df["Contaminates"].value_counts(dropna=False))
print("\nLabel values:")
print(df["Label"].value_counts(dropna=False))

print("\n=== OBJECT SPLITS ===")
print("Split columns:", splits.columns.tolist())
print(splits.head())

# Inspect ZIP images
with zipfile.ZipFile(ZIP_PATH) as archive:
    image_paths = [
        name for name in archive.namelist()
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

print("\n=== IMAGE INFORMATION ===")
print("Total images:", len(image_paths))
print("First five image paths:")
for path in image_paths[:5]:
    print(path)

# Extract numeric image IDs from filenames, e.g.:
# color_img/1000ic_M05H99T0L0C1D0.png
image_ids = []
unmatched_names = []

for path in image_paths:
    filename = Path(path).name
    match = re.match(r"(\d+)ic_", filename)

    if match:
        image_ids.append(int(match.group(1)))
    else:
        unmatched_names.append(filename)

print("\nImage IDs parsed:", len(image_ids))
print("Unique image IDs:", len(set(image_ids)))
print("Unmatched filenames:", len(unmatched_names))

if unmatched_names:
    print("Examples:", unmatched_names[:10])

# Check CSV ID matching
if "ID" in df.columns:
    csv_ids = set(pd.to_numeric(df["ID"], errors="coerce").dropna().astype(int))
    zip_ids = set(image_ids)

    print("\n=== ID MATCHING ===")
    print("CSV unique IDs:", len(csv_ids))
    print("Image unique IDs:", len(zip_ids))
    print("CSV IDs without image:", len(csv_ids - zip_ids))
    print("Image IDs without CSV row:", len(zip_ids - csv_ids))

print("\nInspection complete. No files were modified.")
