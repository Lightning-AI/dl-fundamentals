# EXERCISES

## Exercise 2: Implementing a Custom Dataset Class for Fashion MNIST

In this exercise, we are going to train the multilayer perceptron from Unit 4.3 on a new dataset, [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), based on the PyTorch Dataset concepts introduced in Unit 4.4.

Fashion MNIST is a dataset with a same number of images and image dimension as MNIST. However, instead of handwritten digits, it contains low-resolution images of fashion objects.

![](figures/fashion-mnist.png)



Since the image format of MNIST and Fashion MNIST is identical, we can use the multilayer perceptron code from Unit 4.3 without any modification. The only adjustments we have to make is replacing the MNIST dataset code with a custom `Dataset` class for Fashion MNIST.

To get started, download the GitHub folder [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) and place the `data` subfolder next to the notebook. Then, implement the custom `Dataset` class and train the multilayer perceptron. You should get at least >85% training accuracy.



Hint: You may use the following `Dataset` code as a starter and fill out the missing blanks:

```python
class MyDataset(Dataset):
    def __init__(self, ..., transform=None):

        self.transform = transform
        # ...

    def __getitem__(self, index):
        # ...
        img = torch.tensor(img).to(torch.float32)
        img = img/255.
        # ...

        if self.transform is not None:
            img = self.transform(img)

        # ...

    def __len__(self):
        return self.labels.shape[0]
```



You can use th notebook in this folder as a template: [Unit 4 exercise 2](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit04-multilayer-nets/exercises/2_fashion-mnist)



PS: If you get stuck, please don't hesitate to reach out for help via the forum!