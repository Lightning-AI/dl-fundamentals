# Deep Learning Fundamentals: Code Materials and Exercises



*This repository contains code materials &amp; exercises for Deep Learning Fundamentals course by [Sebastian Raschka](https://sebastianraschka.com) and [Lightning AI](https://lightning.ai).*



- Link to the course website: https://lightning.ai/pages/courses/deep-learning-fundamentals/
- Link to the discussion forum: https://github.com/Lightning-AI/dl-fundamentals/discussions
- Reach out to Lightning & Sebastian on social media: [@LightningAI](https://twitter.com/LightningAI) [@rasbt](https://twitter.com/rasbt)



---

For other announcements, updates, and additional materials, you can follow [Lightning AI](https://twitter.com/LightningAI) and [Sebastian](https://twitter.com/rasbt) on Twitter!

---



## Links to the materials



### Unit 1. Welcome to Machine Learning and Deep Learning [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/unit-1/) ] 

- 1.1 What Is Machine Learning?
- 1.2 How Can We Use Machine Learning?
- 1.3 A Typical Machine Learning Workflow (The Supervised Learning Workflow)
- 1.4 The First Machine Learning Classifier
- 1.5 Setting Up Our Computing Environment
- [1.6 Implementing a Perceptron in Python](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit01-ml-intro/1.6-perceptron-in-python)
- 1.7 Evaluating Machine Learning Models
- Unit 1 exercises
  - [Exercise 1: Add early-stopping to make the Perceptron more efficient](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit01-ml-intro/exercises/1_early-stop)
  - [Exercise 2: Initialize the model parameters with small random numbers instead of 0's](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit01-ml-intro/exercises/2_random-weights)
  - [Exercise 3: Use a learning rate for updating the weights and bias unit](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit01-ml-intro/exercises/3_learning-rate)

### Unit 2. First Steps with PyTorch: Using Tensors [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/2-0-unit-2-overview/) ] 

- 2.1 Introducing PyTorch
- [2.2 What Are Tensors?](https://github.com/Lightning-AI/dl-fundamentals/blob/main/unit02-pytorch-tensors/2.2-tensors/torch-tensors.ipynb)
- [2.3 How Do We Use Tensors in PyTorch?](https://github.com/Lightning-AI/dl-fundamentals/blob/main/unit02-pytorch-tensors/2.3-using-tensors/top10-tensor-commands.ipynb)
- [2.4 Improving Code Efficiency with Linear Algebra](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit02-pytorch-tensors/2.4-linalg)
- [2.5 Debugging Code](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit02-pytorch-tensors/2.5-debugging)
- [2.6 Revisiting the Perceptron Algorithm](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit02-pytorch-tensors/2.6-revisiting-perceptron)
- 2.7 Seeing Predictive Models as Computation Graphs
- Unit 2 exercises
  - [Exercise 1: Introducing more PyTorch functions to make your code more efficient](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit02-pytorch-tensors/exercises/1_torch-where)
  - [Exercise 2: Make the perceptron more efficient using matrix multiplication](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit02-pytorch-tensors/exercises/2_perceptron-matmul)



### Unit 3. Model Training in PyTorch [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/3-0-overview-model-training-in-pytorch/) ] 

- 3.1 Using Logistic Regression for Classification
- 3.2 The Logistic Regression Computation Graph
- 3.3 Model Training with Stochastic Gradient Descent
- 3.4 Automatic Differentiation in PyTorch
- 3.5 The PyTorch API
- [3.6 Training a Logistic Regression Model in PyTorch](https://github.com/Lightning-AI/dl-fundamentals/tree/main/3.6-logreg-in-pytorch)
- 3.7 Feature Normalization
- Unit 3 exercises
  - [Exercise 1: Banknote Authentication](https://github.com/Lightning-AI/dl-fundamentals/tree/main/exercises/1_banknotes)
  - [Exercise 2: Standardization](https://github.com/Lightning-AI/dl-fundamentals/tree/main/exercises/2_standardization) 

### Unit 4. Training Multilayer Neural Networks [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/) ] 

- 4.1 Dealing with More than Two Classes: Softmax Regression
- 4.2 Multilayer Neural Networks and Why We Need Them
- [4.3 Training a Multilayer Perceptron in PyTorch](unit04-multilayer-nets/4.3-mlp-pytorch)
  - [XOR data](unit04-multilayer-nets/4.3-mlp-pytorch/4.3-mlp-pytorch-part1-2-xor)
  - [MNIST data](unit04-multilayer-nets/4.3-mlp-pytorch/4.3-mlp-pytorch-part3-5-mnist)
- [4.4 Defining Efficient Data Loaders](unit04-multilayer-nets/4.4-dataloaders)
- [4.5 Multilayer Neural Networks for Regression](unit04-multilayer-nets/4.5-mlp-regression)
- 4.6 Speeding Up Model Training Using GPUs
- [Unit 4 exercises](./unit04-multilayer-nets/exercises)
  - [Excercise 1: Changing the Number of Layers](./unit04-multilayer-nets/exercises/1_changing-layers)
  - [Exercise 2: Implementing a Custom Dataset Class for Fashion MNIST](./unit04-multilayer-nets/exercises/2_fashion-mnist)

### Unit 5. Organizing your PyTorch Code with Lightning [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/overview-organizing-your-code-with-pytorch-lightning/) ] 

- 5.1 Organizing Your Code with PyTorch Lightning
- [5.2 Training a Multilayer Perceptron in PyTorch Lightning](./unit05-lightning/5.2-mlp-lightning)
- [5.3 Computing Metrics Efficiently with TorchMetrics](./unit05-lightning/5.3-torchmetrics)
- [5.4 Making Code Reproducible](./unit05-lightning/5.4-reproducibility)
- [5.5 Organizing Your Data Loaders with Data Modules](./unit05-lightning/5.5-datamodules)
- [5.6 The Benefits of Logging Your Model Training](./unit05-lightning/5.6-logging)
- [5.7 Evaluating and Using Models on New Data](./unit05-lightning/5.7-evaluating)
- 5.8 Add functionality with callbacks
- [Unit 5 exercises](./unit05-lightning/exercises)

### Unit 6. Essential Deep Learning Tips & Tricks [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/unit-6-overview-essential-deep-learning-tips-tricks/) ] 

- [6.1 Model Checkpointing and Early Stopping](./unit06-dl-tips/6.1-checkpointing)
- [6.2 Learning Rates and Learning Rate Schedulers](./unit06-dl-tips/6.2-learning-rates)
- 6.3 Using More Advanced Optimization Algorithms
- 6.4 Choosing Activation Functions
- [6.5 Automating The Hyperparameter Tuning Process](./unit06-dl-tips/6.5-hparam-opt)
- 6.6 Improving Convergence with Batch Normalization
- 6.7 Reducing Overfitting With Dropout
- [6.8 Debugging Deep Neural Networks](./unit06-dl-tips/6.8-debugging)
- [Unit 6 exercises](./unit06-dl-tips/exercises)


### Unit 7. Getting Started with Computer Vision [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/unit-7-overview-getting-started-with-computer-vision/) ] 

- 7.1 Working With Images
- 7.2 How Convolutional Neural Networks Work
- 7.3 Convolutional Neural Network Architectures
- [7.4 Training Convolutional Neural Networks](./unit07-computer-vision/unit07-computer-vision/7.4-cnn-training)
- [7.5 Improving Predictions with Data Augmentation](./unit07-computer-vision/unit07-computer-vision/)
- [7.6 Leveraging Pre-trained Models with Transfer Learning](./unit07-computer-vision/unit07-computer-vision/)
- [7.7 Using Unlabeled Data with Self-Supervised](./unit07-computer-vision/unit07-computer-vision/)
- [Unit 7 exercises](./unit07-computer-vision/exercises)



### Unit 8. Introduction to Natural Language Processing and Large Language Models  [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/unit-8.0-natural-language-processing-and-large-language-models/) ] 

- 8.1 Working with Text Data
- [8.2 Training Text Classifier Baseline](unit08-large-language-models/8.2-bag-of-words)
- 8.3. Introduction to Recurrent Neural Networks
- 8.4 From RNNas to the Transformer Architecture
- 8.5 Understanding Self-Attention
- 8.6 Large Language Models
- [8.7 Using Large Language Model for Classification](unit08-large-language-models/8.7-distilbert-finetuning)
- [Unit 8 exercises](unit08-large-language-models/exercises)

### Unit 9. Techniques for Speeding Up Model Training  [ [Link to videos](https://lightning.ai/pages/courses/deep-learning-fundamentals/9.0-overview-techniques-for-speeding-up-model-training/) ] 

- [9.1 Accelerated Model Training via Mixed-Precision Training](unit09-performance/9.1-mixed-precision)
- 9.2 Multi-GPU Training Strategies
- [9.3 Deep Dive Into Data Parallelism](unit09-performance/9.3-multi-gpu)
- [9.4 Compiling PyTorch Models](unit09-performance/9.4-compile)
- [9.5 Increasing Batch Sizes to Increase Throughput](unit09-performance/9.5-batchsize-finder)
- [Unit 9 exercises](unit09-performance/exercises)

### Unit 10. Coming Soon
