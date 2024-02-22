# text_gen

A basic CLI transformer model to generate arbitrary text. 
**This project is for my own educational purposes only and not intended for any production application**

Based on the paper "Attention is All You Need" (Vaswani, et al., [arXiv:1706.03762]<https://arxiv.org/abs/1706.03762>) and the [nanoGPT]<https://github.com/karpathy/nanoGPT> implementation by Andrej Karpathy.

The model itself was too large to upload to GitHub, however it was trained on approximately 12GB of text from the "UltraTextbooks" dataset hosted on the [HuggingFace]<https://huggingface.co/datasets/Locutusque/UltraTextbooks> platform. Training took about 12 hours on an Nvidia RTX 4090 GPU and achieved a validation loss around 2.5.

![](https://github.com/bbehnkeSE/text_gen/blob/main/assets/train_val_loss.png)

Generate.py is used to generate the arbitrary text to either a file or the standard console (example_output.txt shows an example output of 1,000 tokens). Generate.py is able to accept a string as a command line argument to offer context to the generation, though it is unable to respond to queries in any appreciable way.