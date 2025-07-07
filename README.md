# Neural-Network


Intro to Deep Learning assignment:

## Multi-Layer Perceptron

The first part of this assignment is to implement from scratch a **Multi-Layer Perceptron** that supports an arbitrary number of hidden layers, each with an arbitrary number of units **using Numpy only** (including gradient calculations)

* For this, I implemented activations, loss functions, batch normalization, forward and backward methods with momentum for linear layers, and all of the aforementioned derivatives

>  [mytorch](https://github.com/yijing-sie/Neural-Network/tree/main/mytorch) is my custom deep learning library, built entirely in NumPy, that functions similarly to established DL lebraries like PyTorch or TensorFlow
*  [MLP.py](MLP.py) contains my implementation of Multi-Layer Perceptron

## Neural Netwrok

For the second part, the goal is to employ feedward neural network to predict phoneme state labels for Mel spectrogram frames in the test set, a task known as speech recognition.

* The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text
* **Training features**: Raw mel spectrogram frames
* **Training labels**: Frame-level phoneme state labels

**Note**: The length of a training label array per frame will be as long as however many phonemes are in the utterance.
* It is also a kaggle competition, and all the details can be found [here](https://www.kaggle.com/competitions/11785-homework-3-part-2-slack-seq-to-seq)
* My model achieves **0.80759** for unweighted frame-level accuracy on phoneme state labels in the `test` set
* It's ranked number **5** in a class of 300+ students
* All the work for this part can be found in [NN_speach_recognition.ipynb](NN_speach_recognition.ipynb)
* I provided my model with “context” of size 25 around each mel spectrogram frame for better performance





