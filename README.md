# Neural-Network


Intro to Deep Learning assignment:

## Multi-Layer Perceptron

The first part of this assignment is to implement from scratch a numpy-based **Multi-Layer Perceptron** that supports an arbitrary number of hidden layers, each with an arbitrary number of units **without using Pytorch**

> For this, I implemented activations, loss fucntions, batch normalization, forward and backward methods with momentum for linear layers, and all of the aforementioned derivatives

*  [mytorch](https://github.com/yijing-sie/Neural-Network/tree/main/mytorch) is my own custom deep learning library, which acts similar to other deep learning libraries like PyTorch or Tensorflow, contains all files needed for MLP.py
*  [MLP.py](MLP.py) contains my implementation of Multi-Layer Perceptron

## Neural Netwrok

For the second part, the goal is to employ feedward neural network to predict phoneme state labels for Mel spectrogram frames in the test set, a task known as speech recognition.

* The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text
* **Training features**: Raw mel spectrogram frames
* **Training labels**: Frame-level phoneme state labels

**Note**: The length of a training label array per frame will be as long as however many phonemes are in the utterance.
* It is also a kaggle competition, and all the details can be found [here](https://www.kaggle.com/competitions/idl-fall2021-hw1p2/overview)
* All the work for this part can be found in [NN_speach_recognition.ipynb](NN_speach_recognition.ipynb)
* I provided my model with “context” of size K around each mel spectrogram frame for better performance
* My model achieves **0.80759** for unweighted frame-level accuracy on phoneme state labels in the `test` set




