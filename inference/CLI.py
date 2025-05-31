"""Simple CLI chatbot powered by TestLLM.

This script loads a SentencePiece tokenizer model and the ``testllm.pth``
checkpoint to provide a minimal command line interface for text generation.
It is intentionally simple and is only meant to demonstrate that the model and
vocabulary can be used for inference. The tokenizer's vocabulary includes the
following special tokens:
```
<unk> <bos> <eos> <start> <end> <reason> <answer> <agent> <context> <sep>
```
"""

import argparse
import torch
import sentencepiece as spm
from inference.sampler import Sampler, GreedySampler, TopKSampler, MinPSampler
from src.models.testllm import TestLLM


def load_model(
    model_path: str, tokenizer: spm.SentencePieceProcessor, device: torch.device
) -> TestLLM:
    """Load ``TestLLM`` model weights."""
    params = {
        "vocab_size": tokenizer.vocab_size(),
        "embed_size": 512,
        "num_mha": 4,
        "num_heads": 16,
        "ff_size": 1024,
    }
    model = TestLLM(**params)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def generate(
    model: TestLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    sampler: Sampler = GreedySampler(),
    temperature: float = 1.0,
) -> str:
    """Generate text from ``prompt`` using greedy decoding."""
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    input_ids = [bos_id] + tokenizer.encode(prompt, out_type=int)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        mask = input_tensor == 0  # <pad> token id is 0 when training the tokenizer
        with torch.no_grad():
            logits = model(input_tensor, mask)
        logits = logits[0, -1]  # Get logits for the last token
        next_id = sampler.sample(logits, temperature=temperature)
        input_tensor = torch.cat(
            [input_tensor, torch.tensor([[next_id]], device=device)], dim=1
        )
        if next_id == eos_id:
            print("Generated EOS token, stopping generation.")
            break
    generated_ids = input_tensor.squeeze().tolist()[len(input_ids) :]
    return tokenizer.encode_as_string(generated_ids)


def chat(
    model: TestLLM,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int,
    sampler: Sampler,
) -> None:
    """Run an interactive chat loop."""
    print('Type "quit" or "exit" to stop.')
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        response = generate(model, tokenizer, prompt, device, max_new_tokens, sampler)
        print(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with TestLLM on the command line"
    )
    parser.add_argument(
        "--model-path",
        default="src/models/testllm.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="tokenizer/tokenizer.model",
        help="Path to the SentencePiece tokenizer model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--sampler", 
        choices=["greedy", "topk", "minp"],
        default="greedy",
        help="Sampling strategy to use for text generation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = GreedySampler() if args.sampler == "greedy" else \
                TopKSampler(k=50) if args.sampler == "topk" else \
                MinPSampler(ratio=0.1)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    model = load_model(args.model_path, tokenizer, device)
    chat(model, tokenizer, device, args.max_new_tokens, sampler)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
