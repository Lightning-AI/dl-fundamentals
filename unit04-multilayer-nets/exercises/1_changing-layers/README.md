# EXERCISES



## Exercise 1: Changing the Number of Layers

In this exercise, we are toying around with the multilayer perceptron architecture from Unit 4.3.

In Unit 4.3, we fit the following multilayer perceptron on the MNIST dataset:

```python
class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits
```

This network had 40,785 parameters (see Quiz on how to calculate this number), and it achieved the following accuracy values:

- Train Acc 97.24%
- Val Acc 95.64%
- Test Acc 96.46%



Can you change the architecture to achieve the same (or better) performance with fewer parameters and only 1 hidden layer?



PS: You may also try to add additional layers, but as a rule of thumb, using more than two hidden layers in a multi-layer perceptron rarely improves the predictive performance.



You can use th notebook in this folder as a template: [Unit 4 exercise 1](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit04-multilayer-nets/exercises/1_changing-layers)