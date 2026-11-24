
from pathlib import Path
import pandas as pd
import zipfile
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "contamination_dataset"

df = pd.read_csv(DATA / "dataset.csv")
splits = pd.read_csv(DATA / "object_splits.csv")

# Normalize names for matching
df["item_key"] = df["Item"].astype(str).str.strip().str.casefold()
splits["item_key"] = (
    splits["item_name"].astype(str).str.strip().str.casefold()
)

print("=== LABEL DISTRIBUTION ===")
print(df["Contaminates"].value_counts(dropna=False))

print("\n=== SPLIT DISTRIBUTION ===")
print(splits["split"].value_counts(dropna=False))

print("\n=== ITEM MATCHING ===")
split_names = set(splits["item_key"])
matched = df["item_key"].isin(split_names)

print("Dataset rows:", len(df))
print("Rows matching split item names:", int(matched.sum()))
print("Rows without a matching split:", int((~matched).sum()))

if (~matched).any():
    print("\nExamples without matching split:")
    print(df.loc[~matched, ["ID", "Item"]].head(15).to_string(index=False))

print("\n=== SAMPLE LABELS BY MATERIAL ===")
print(pd.crosstab(df["Material"], df["Contaminates"]).to_string())

print("\n=== IMAGE FILE CHECK ===")
with zipfile.ZipFile(DATA / "color_img.zip") as z:
    image_names = [
        Path(n).name for n in z.namelist()
        if n.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

image_ids = {
    int(match.group(1))
    for name in image_names
    if (match := re.match(r"(\d+)ic_", name))
}

csv_ids = set(pd.to_numeric(df["ID"], errors="coerce").dropna().astype(int))
print("Image IDs:", len(image_ids))
print("CSV IDs:", len(csv_ids))
print("Missing image IDs:", len(csv_ids - image_ids))
print("\nInspection complete. No files were modified.")




from pathlib import Path
import pandas as pd
from PIL import Image

base_dir = Path("contamination_dataset")
df = pd.read_csv(base_dir / "dataset.csv")
image_dir = base_dir / "color_img"


print("\n=== SAMPLE CONTAMINATION RECORDS ===")

import re
from pathlib import Path

# Search recursively in case the images are nested in another folder.
image_files = list(image_dir.rglob("*.png"))

# Build a lookup using the numeric prefix before "ic_".
image_lookup = {}
for path in image_files:
    match = re.match(r"(\d+)ic_", path.name)
    if match:
        image_lookup[int(match.group(1))] = path

print("PNG images discovered:", len(image_files))
print("Images indexed by ID:", len(image_lookup))

sample = df[df["Contaminates"] == "Yes"].head(10)

for _, row in sample.iterrows():
    image_id = int(row["ID"])
    image_path = image_lookup.get(image_id)

    print("\nImage ID:", image_id)
    print("Item:", row["Item"])
    print("Material:", row["Material"])
    print("Contaminates:", row["Contaminates"])
    print("Label:", row["Label"])
    print("Image path:", image_path)
    print("Image found:", image_path is not None)




from PIL import Image, ImageDraw
from pathlib import Path
import math

# Show 12 images labeled Contaminates = Yes
sample = df[df["Contaminates"] == "Yes"].head(12)

columns = 3
cell_w, cell_h = 320, 280
rows = math.ceil(len(sample) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_w, rows * cell_h),
    "white"
)
draw = ImageDraw.Draw(sheet)

for index, (_, row) in enumerate(sample.iterrows()):
    image_path = image_lookup.get(int(row["ID"]))

    if image_path is None:
        continue

    with Image.open(image_path) as original:
        image = original.convert("RGB")
        image.thumbnail((300, 220))

    x = (index % columns) * cell_w
    y = (index // columns) * cell_h

    sheet.paste(image, (x + 10, y + 10))
    draw.text(
        (x + 10, y + 235),
        f'ID: {row["ID"]} | {row["Material"]}',
        fill="black"
    )
    draw.text(
        (x + 10, y + 253),
        f'Contaminates: {row["Contaminates"]} | Label: {row["Label"]}',
        fill="black"
    )

output_path = Path("contamination_sample.jpg")
sheet.save(output_path)

print(f"\nVisual sample saved to: {output_path.resolve()}")
