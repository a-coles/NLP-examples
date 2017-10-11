# A cardinal vowel classifier using a neural net

This script uses F1 and F2 vowel data to train a multi-class classifier that can classify three cardinal vowels: iy, ah, and uw. These vowels make up three corners of the standard vowel quadrilateral. This example was adapted from the binary classifier written by Tim O'Donnell and was developed using vowel formant measurements from the Hillenbrand dataset.

This model trains and tests a dense deep network using rectified linear units as the intermediate activation functions and softmax for the final classification activation function. The model uses the Adam optimizer function, as through testing, this was found to give better results than stochastic gradient descent. The loss function this model uses is categorical cross-entropy.

## Prerequisites

* Numpy
* Scipy
* matplotlib
* pandas
* keras

## Usage

This script can be run directly from the command line with no options. Ensure that the vowel data .csv file is in the same directory as the script, and run like this:

```python3 classify_cardinal_vowels.py```