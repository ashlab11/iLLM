# sample_slimpajama_chunk.py
"""
Stream a random SlimPajama chunk (10 % of corpus) until ~10 GiB of text.
"""

import random, os, json, requests, io, zstandard as zstd
import sentencepiece as spm
from tqdm import tqdm
tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer.model")

CHUNK_ID   = random.randint(0, 9)        # choose chunk0 … chunk9 uniformly
SAMPLE_GB  = 10
OUTFILE    = f"data/slimpajama_chunk{CHUNK_ID}_10G.txt"
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

def shard_url(idx: int) -> str:
    return (f"https://huggingface.co/datasets/cerebras/SlimPajama-627B/"
            f"resolve/main/train/chunk{CHUNK_ID}/"
            f"example_train_{idx}.jsonl.zst")

# SlimPajama lists ~1 000 shards per chunk
file_urls = [shard_url(i) for i in range(1000)]

def stream_zst(url):
    """Yield JSON objects from a .jsonl.zst HuggingFace file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        dctx   = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(r.raw)          # stream‑decompress
        textio = io.TextIOWrapper(reader, encoding="utf-8")
        for line in textio:
            yield json.loads(line)

# ── stream shards in random order ────────────────────────────────────────────
written = 0
with open(OUTFILE, "w", encoding="utf-8") as fout, \
    tqdm(total=SAMPLE_GB * 1024**3, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"chunk{CHUNK_ID}  ➜ writing",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit}  {postfix}") as pbar:
    
    for url in random.sample(file_urls, len(file_urls)):
        try:
            for ex in stream_zst(url):
                text = ex["text"].strip()
                clean = " ".join(text.split())
                if len(tokenizer.encode_as_ids(clean)) < 500:  # skip too short examples
                    continue
                fout.write(clean + "\n")
                delta = len(clean.encode())
                written += delta
                pbar.update(delta)
                pbar.set_postfix_str(f"{written/1024**3:.1f} GiB")

                if written >= SAMPLE_GB * 1024**3:
                    raise StopIteration
        except StopIteration:
            break
        except Exception as e:
            print("Skipping shard:", url, "error:", e)

print(f"Wrote ~{written/1024**3:.1f} GiB to {OUTFILE} (chunk{CHUNK_ID})")
