# scripts/sample_slimpajama_retry.py
import os
import time
from datasets import load_dataset, DownloadConfig
from requests.exceptions import HTTPError

OUT_PATH = "data/slimpajama_sample.txt"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
TARGET_BYTES = 10 * 1024**3  # 10 GiB

download_config = DownloadConfig(
    token=True,  # authenticate requests
    max_retries=5
)

ds = load_dataset(
    "MBZUAI-LLM/SlimPajama-627B-DC",
    split="train",
    streaming=True,
    download_config=download_config,
)

written = 0
with open(OUT_PATH, "a", encoding="utf-8") as f:
    for ex in ds:
        text = ex.get("text") or ex.get("excerpt") or ""
        line = text.strip() + "\n"
        if not line.strip():
            continue

        # try writing, with simple back-off on HTTPError
        for attempt in range(3):
            try:
                f.write(line)
                written += len(line.encode("utf-8"))
                break
            except HTTPError as e:
                print(f"[Attempt {attempt+1}] HTTPError: {e}. Backing off 10s…")
                time.sleep(10)
        else:
            # if still failing after retries, skip this example
            print("Skipping after 3 failed attempts.")
            continue

        # stop once we hit our target
        if written >= TARGET_BYTES:
            print(f"Reached ~{written/1024**3:.1f} GiB, stopping.")
            break

print("Finished sampling SlimPajama-DC.")
