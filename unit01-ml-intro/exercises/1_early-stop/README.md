# EXERCISES

## Exercise 1: Add early-stopping to make the Perceptron more efficient.

In its original implementation, the perceptron completes the number of epochs specified via the `epochs` argument:

```python
def train(model, all_x, all_y, epochs):
    ...
```

However, this can result in executing too many unnecessarily. Modify the `train` function in Section 4 (`4) Implementing the Perceptron` such that it automatically stops when the perceptron classifies the training data perfectly.