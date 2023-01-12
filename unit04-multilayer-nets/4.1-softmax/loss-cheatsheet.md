# PyTorch Loss Function Cheatsheet

PyTorch Loss-Input Confusion (Cheatsheet)

- [`torch.nn.functional.binary_cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html) takes logistic sigmoid values as inputs
- [`torch.nn.functional.binary_cross_entropy_with_logits`](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) takes logits as inputs
- [`torch.nn.functional.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) takes logits as inputs (performs log_softmax internally)
- [`torch.nn.functional.nll_loss`](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html) is like `cross_entropy` but takes log-probabilities (log-softmax) values as inputs