# Deep Learning Fundamentals Unit 5

## Organizing your Code with PyTorch Lightning

In Unit 5, we introduce the PyTorch Lightning Trainer, which helps us organize our PyTorch code and take care of lots of the mundane boilerplate code.

In particular you will learn how to …

- organize our PyTorch code with Lightning;
- compute metrics efficiently with TorchMetrics;
- make our code reproducible;
- organize our data loaders via DataModules;
- logging results during training;
- adding extra functionality with callbacks.

This Studio provides a reproducible environment with the supplementary code for Unit 5 of the [**Deep Learning Fundamentals**](https://lightning.ai/pages/courses/deep-learning-fundamentals/) class by Sebastian Raschka, which is freely available at Lightning AI.

<br>

**What's included?**

Click the "Run Template" button at the top of this page to launch into a Studio environment that contains the following materials:

- `code-units/`:

  - `5.2-mlp-lightning`: training a multilayer perceptron using the PyTorch Lightning Trainer

  - `5.3-torchmetrics`: computing metrics efficiently with TorchMetrics

  - `5.4-reproducibility`: making code reproducible
  
  - `5.5-datamodules`: organizing your data loaders with data modules
  
  - `5.6-logging`:  the benefits of logging your model training
  
  - `5.7-evaluating`: Evaluating and using models on new data


- `exercises/`: 
  - `1_lightning-regression`: Exercise 1,  changing a classifier to a regression model
  - `2_custom-callback`: Exercise 2, developing a custom plugin for tracking training and validation accuracy
- `solutions/`: Solutions to the exercises above

---

<br>

<iframe width="560" height="315" src="https://www.youtube.com/embed/DxALtmlxQ4U?si=Qa8hF9NPVdQRRMVf" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

- Videos of [Part 2](https://www.youtube.com/watch?v=Y11-leJtC1k&list=PLaMu-SDt_RB7QZz-kTjdE_IkNDG0ZcHr-&index=4) and [Part 3](https://www.youtube.com/watch?v=8NKXArrnJlQ&list=PLaMu-SDt_RB7QZz-kTjdE_IkNDG0ZcHr-&index=5)
- [The complete YouTube Playlist](https://www.youtube.com/watch?v=x4UvpMsyG8M&list=PLaMu-SDt_RB7QZz-kTjdE_IkNDG0ZcHr-) with all 19 videos in Unit 5
- [Or access the Unit 5 videos on the Lightning website](https://lightning.ai/courses/deep-learning-fundamentals/), which includes additional quizzes

<br>

## About Unit 5: Organizing your Code with PyTorch Lightning

The previous units focused on learning how deep neural networks work from scratch. Along the way, we introduced PyTorch in units 2 and 3, and we trained our first multilayer neural networks in Unit 4. Personally, I really like PyTorch’s balance between customizability and user-friendliness.

However, as we start working with more sophisticated features, including model checkpointing, logging, multi-GPU training, and distributed computing, PyTorch can sometimes be a bit too verbose. Hence, in this unit, we will introduce the Lightning Trainer, which helps us organize our PyTorch code and take care of lots of the mundane boilerplate code.
