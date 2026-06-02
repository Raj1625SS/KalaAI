from ddgs import DDGS
import requests
import os
import time
import random
from pathlib import Path

# Multiple queries per class to get more variety
classes = {
    "ajrakh": [
        "ajrakh block print fabric texture",
        "ajrakh print textile India",
        "ajrakh fabric pattern closeup",
        "ajrakh hand block print cotton",
        "ajrakh resist print indigo fabric",
    ],
    "kalamkari": [
        "kalamkari block print fabric texture",
        "kalamkari textile pattern India",
        "kalamkari printed cotton fabric",
        "kalamkari hand block print closeup",
        "kalamkari fabric design pattern",
    ],
    "dabu": [
        "dabu mud resist print fabric",
        "dabu print rajasthan textile",
        "dabu hand block print fabric",
        "dabu resist dyeing fabric pattern",
        "dabu printed cotton India",
    ],
    "bengal_hand_block": [
        "bengal hand block print cotton fabric",
        "bengal block print textile pattern",
        "west bengal hand printed fabric",
        "bengal block print saree texture",
        "bengal hand printed cotton closeup",
    ],
}

def download_images(folder, queries, max_per_query=50):
    Path(folder).mkdir(parents=True, exist_ok=True)
    downloaded = 0
    existing = set(os.listdir(folder))

    for query in queries:
        print(f"  Query: '{query}'")
        try:
            time.sleep(random.uniform(4, 8))  # avoid rate limit between queries
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_per_query))

            for i, result in enumerate(results):
                try:
                    url = result["image"]
                    response = requests.get(url, timeout=8)
                    if response.status_code == 200:
                        ext = url.split(".")[-1].split("?")[0]
                        if ext.lower() not in ["jpg", "jpeg", "png", "webp"]:
                            ext = "jpg"
                        filename = f"{downloaded:04d}.{ext}"
                        if filename not in existing:
                            filepath = os.path.join(folder, filename)
                            with open(filepath, "wb") as f:
                                f.write(response.content)
                            downloaded += 1
                            existing.add(filename)
                            print(f"  [{downloaded}] saved", end="\r")
                except Exception:
                    continue
                time.sleep(0.1)

        except Exception as e:
            print(f"  Skipping query due to error: {e}")
            continue

    print(f"\n  Total saved: {downloaded} images")
    return downloaded

for class_name, queries in classes.items():
    print(f"\nDownloading: {class_name}...")
    folder = f"data/train/{class_name}"
    count = download_images(folder, queries, max_per_query=50)
    print(f"  Waiting before next class...")
    time.sleep(random.uniform(8, 12))  # longer wait between classes

print("\nAll downloads complete!")