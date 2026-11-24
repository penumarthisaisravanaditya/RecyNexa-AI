
import json
from pathlib import Path
from PIL import Image, ImageDraw

data_dir = Path("packwise_dataset/data")
json_path = data_dir / "train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
image_dir = data_dir / "train"

canvas = Image.new("RGB", (1200, 900), "white")
draw = ImageDraw.Draw(canvas)

thumb_w, thumb_h = 280, 250

for index, item in enumerate(images[:12]):
    filename = item["file_name"]

    # Try the filename from the annotation first.
    candidates = [
        image_dir / filename,
        image_dir / Path(filename).name,
    ]

    image_path = next(
        (p for p in candidates if p.is_file()),
        None,
    )

    if image_path is None:
        print("Image not found:", filename)
        continue

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((thumb_w, thumb_h))

        x = (index % 4) * 300
        y = (index // 4) * 300

        canvas.paste(img, (x, y))
        draw.text(
            (x + 5, y + 255),
            Path(filename).name[:40],
            fill="black",
        )

output = Path("packwise_sample.jpg")
canvas.save(output)
print("Saved sample:", output.resolve())
