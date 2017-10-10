# A simple positive-negative classifier

This script takes as input lists of statements rated "positive" and "negative", splits them into a training and a testing set, trains a classifier on the training data, and shows the results of attempts to classify the statements in the testing set.

## Prerequisites

* Numpy
* Scipy
* matplotlib
* sklearn
* nltk

## Usage

### Basic usage

First, whatever positive and negative data you want to use must be preprocessed. (This example was developed using the already-processed movie review dataset used by Pang & Lee, 2005.) If you would like to use your own, you should split the positive and negative sentences into their own files, with each sentence on its own line. All punctuation ('.', ',', etc.) should be padded with a space on each side.

The script can be run as such:

```python classify_reviews.py pos/file/path neg/file/path vectorize_method classify_method```

The included possible vectorization methods are:

* `unigram` - for unigrams,
* `bigram` - for bigrams,
* `both` - for both unigrams and bigrams.

The included possible classification methods are:

* `logreg` - for logistic regression,
* `svm` - for support vector machine (with a linear kernel),
* `nb` - for Naive Bayes,
* `dummy` - to allocate labels with equal probability.

### Options

Included optional switches are:

* `-sw` or `--stopwords` - use nltk's built-in list of stopwords for English (to ignore these common words in training),
* `-cm` or `--confusionmatrix` - print out and show a figure of the resulting confusion matrix.