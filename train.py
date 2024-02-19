import os
import sys
import json

import torch
import torch.nn  as nn
import tiktoken  as tik

from   model     import Decoder, CharTokenizer
from   torch.nn  import functional as F
from   functools import partial
from   datetime  import datetime
from   losses    import training_loss, validation_loss

# Hyperparameters
BATCH_SIZE    = 32  # Number of independent sequences we will process in parallel
BLOCK_SIZE    = 512   # Max context length for predictions
MAX_ITERS     = 10001
EVAL_ITERS    = 200
EVAL_INTERVAL = 1000
L_RATE        = 6e-4
N_EMBED       = 384  # Number of embedding dimensions
NUM_HEADS     = 6
HEAD_SIZE     = N_EMBED // NUM_HEADS
N_LAYER       = 6
DROPOUT       = 0.2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MB            = 1_048_576
GB            = 1_073_741_824


# def load_data(path, amount=-1):
#     with open(path, 'r', encoding="utf-8") as f:
#         data = f.read(amount)

#     return data


def train_test_split(data_enc, percent=0.9):
    n = int(percent * len(data_enc))
    train = data_enc[:n]
    val   = data_enc[n:]

    return train, val


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([data[i:i + BLOCK_SIZE]         for i in ix])
    y  = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()

    losses_train = torch.zeros(EVAL_ITERS)
    losses_test  = torch.zeros(EVAL_ITERS)
    for i in range(EVAL_ITERS):
        X_train, Y_train = get_batch(train_data)
        X_test,  Y_test  = get_batch(val_data)

        _, train_loss = model(X_train, Y_train)
        _, test_loss  = model(X_test, Y_test)

        losses_train[i] = train_loss.item()
        losses_test[i]  = test_loss.item()

    out["train"] = losses_train.mean()
    out["val"]   = losses_test.mean()

    model.train()

    return out


def main():
    # Get tokens
    token = tik.get_encoding("p50k_base")
    vocab_size = token.n_vocab

    # Creating or loading model
    print(f"Creating model and moving it to {'gpu' if DEVICE == 'cuda' else 'cpu'}...")
    model = Decoder(vocab_size, BLOCK_SIZE, N_EMBED, HEAD_SIZE, NUM_HEADS, N_LAYER, DROPOUT, DEVICE)

    try:
        model.load_state_dict(torch.load(f"model/model.pt"))
        print(f"model/model.pt loaded.")
    except:
        print(f"model.pt could not be loaded, new file will be created.")

    
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=L_RATE)
    print(f"Moved model to {'gpu' if next(model.parameters()).is_cuda else 'cpu'}.")

    model.train()

    print("Loading an absurd amount of data...")
    data = []
    with open("datasets/ultra_textbooks/ut_corpus.txt", 'rb') as f:
        for chunk in iter(partial(f.read, GB//2), b''):
            data.append(chunk)

    print("Done.")

    best_val   = 2.6418
    total_data = len(data)
    # Training loop
    print(f"Training on {'gpu' if next(model.parameters()).is_cuda else 'cpu'}...")  
    for data_num, d in enumerate(data):
        print(f"\nBegin dataset {data_num+1} of {total_data}")
        train_data, val_data = train_test_split(torch.tensor(token.encode(d.decode()), dtype=torch.long))

        for i in range(MAX_ITERS):
            # Get a batch of the data
            xb, yb = get_batch(train_data)

            # Eval noisy loss on batch
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
            # Evaluate loss on train and val sets
            if i % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, train_data, val_data)
                print(f"Step {i}: Train loss - {losses['train']:.4f}. Val loss {losses['val']:.4f}. ({datetime.now():%H:%M %m/%d/%y})")
                training_loss.append(float(losses["train"]))
                validation_loss.append(float(losses["val"]))

                with open("losses.py", 'w') as l:
                    l.write(f"{training_loss   = }\n")
                    l.write(f"{validation_loss = }\n")

                # If validation loss is better than current best, update model
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    torch.save(model.state_dict(), "model/model.pt")

                    print(f"\n!---- Updated model. New best validation score: {best_val:.4f}. ----!\n")

                # Save a checkpoint every evaluation
                torch.save(model.state_dict(), f"model/checkpoints/checkpoint_{datetime.now():%H_%M_%m_%d_%y}.pt")


if __name__ == "__main__":
    main()