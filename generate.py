from model import Decoder
import torch
import tiktoken as tik

import sys

# Hyperparameters
BATCH_SIZE    = 32  # Number of independent sequences we will process in parallel
BLOCK_SIZE    = 512   # Max context length for predictions
MAX_ITERS     = 10001
EVAL_ITERS    = 200
EVAL_INTERVAL = 1000
L_RATE        = 6e-5
N_EMBED       = 384  # Number of embedding dimensions
NUM_HEADS     = 6
HEAD_SIZE     = N_EMBED // NUM_HEADS
N_LAYER       = 6
DROPOUT       = 0.2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MB            = 1_048_576


def main():
    if len(sys.argv) < 2:
        prompt = '\n'
    else:
        prompt = sys.argv[1]

    token = tik.get_encoding("p50k_base")
    vocab_size = token.n_vocab
    model = Decoder(vocab_size, BLOCK_SIZE, N_EMBED, HEAD_SIZE, NUM_HEADS, N_LAYER, DROPOUT, DEVICE)
    model.to(DEVICE)

    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()

    with open("example_output.txt", 'w', encoding="utf-8") as f:
        f.write(token.decode(model.generate(torch.tensor(
            (token.encode(prompt),), dtype=torch.long, device=DEVICE), 1_000)[0].tolist()))

    # print(token.decode(model.generate(torch.tensor(
    #     (token.encode(prompt),), dtype=torch.long, device=DEVICE), 1_000)[0].tolist()))


if __name__ == "__main__":
    main()
