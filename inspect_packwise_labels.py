
import json
from pathlib import Path

data_dir = Path("packwise_dataset/data")

for split in ["train", "val", "test"]:
    json_path = data_dir / f"{split}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    print(f"\n=== {split.upper()} ===")
    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", len(categories))

    for category in categories:
        print(
            f'ID: {category["id"]} | '
            f'Name: {category["name"]}'
        )
