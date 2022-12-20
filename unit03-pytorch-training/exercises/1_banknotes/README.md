# EXERCISES

## Exercise 1: Banknote Authentication

In this exercise, we are applying logistic regression to a banknote authentication dataset to distinguish between genuine and forged bank notes.


**The dataset consists of 1372 examples and 4 features for binary classification.** The features are 

1. variance of a wavelet-transformed image (continuous) 
2. skewness of a wavelet-transformed image (continuous) 
3. kurtosis of a wavelet-transformed image (continuous) 
4. entropy of the image (continuous) 

(You can fine more details about this dataset at [https://archive.ics.uci.edu/ml/datasets/banknote+authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).)


In essence, these four features represent features that were manually extracted from image data. Note that you do not need the details of these features for this exercise. 

However, you are encouraged to explore the dataset further, e.g., by plotting the features, looking at the value ranges, and so forth. (We will skip these steps for brevity in this exercise)

Link to exercise notebook: [exercise_1_banknotes.ipynb](https://github.com/Lightning-AI/dl-fundamentals/blob/main/unit03-pytorch-training/exercises/1_banknotes/exercise_1_banknotes.ipynb)