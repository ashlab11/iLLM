import sentencepiece as spm
import os

# Paths and parameters
CORPUS_FILE   = "data/slimpajama_sample.txt"
MODEL_DIR     = "tokenizer"
MODEL_PREFIX  = os.path.join(MODEL_DIR, "tokenizer")
VOCAB_SIZE    = 16000
CHAR_COVERAGE = 1.0 
MAX_SENTENCE_LENGTH = 10000
SAMPLE_SIZE = 2500000  

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Define special tokens
special_tokens = [
    "<start>", "<end>", "<reason>", "<answer>",
    "<agent>", "<context>", "<sep>"
]

# Train the SentencePiece Unigram tokenizer
spm.SentencePieceTrainer.Train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type='unigram',
    character_coverage=CHAR_COVERAGE,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<bos>',
    eos_piece='<eos>',
    user_defined_symbols=special_tokens,
    max_sentence_length=MAX_SENTENCE_LENGTH,
    input_sentence_size=SAMPLE_SIZE,
    shuffle_input_sentence=True
)

print(f"Trained tokenizer saved as {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")
