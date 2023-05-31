##  # Exercise 1: Using Fabric

It's been a while since we used pure PyTorch code in Unit 4. If you are interested in using Fabric, a good exercise is to convert the existing PyTorch code into Fabric -- it only requires changing a handful of lines.

For this, you can use [this template](exercise_1-template.ipynb) based on the MNIST classifier from Unit 4.

Remember, the conversion to Fabric involves

1. Setting up the model and optimizer via `fabric.setup(model, optimizer)`
2. Converting the data loaders via `fabric.setup_dataloaders`

Hint: If you prefer to run this code on a CPU instead of GPU, you don't need to specify the accelerator. Or just set it to `accelerator="cpu"`.