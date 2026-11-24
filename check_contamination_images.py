
from pathlib import Path

root = Path("contamination_data")
extensions = {".jpg", ".jpeg", ".png"}

for folder_name in ["clean", "contaminated"]:
    folder = root / folder_name
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"{folder_name}: {len(images)} images")

total = sum(
    1 for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

print(f"Total image files: {total}")
