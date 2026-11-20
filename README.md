# tinygpt-from-scratch

A small GPT written from scratch in pure PyTorch (no `nn.Transformer`, no HuggingFace) and trained on CPU on a TinyStories subset.
The experiment asks **"Does the tokenizer matter?"**: char-level vs our own byte-level BPE (vocab 512 and 2048), plus a no-positional-embedding ablation.
Every run gets the same budget — 7 minutes of training on 4 CPU threads — and runs are compared by validation **bits per character** on identical text.
A separate roofline study places each of the model's ops against this machine's compute and memory-bandwidth ceilings.
See `SPEC.md` for the full contract, `RESULTS.md` for findings, and `demo.ipynb` to try it in Google Colab.
