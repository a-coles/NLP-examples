import scipy
import numpy as np
import itertools
import sys
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

# Usage:
# python classify_reviews.py pos/file/path neg/file/path 
#							vectorize_method classify_method stopwords
# Possible vectorize methods are:
#	unigram - for unigrams
#	bigram - for bigrams
#	both - for both unigrams and bigrams
# Possible classifier methods are:
#	logreg - for logistic regression
#	svm - for support vector machine (with a linear kernel)
#	nb - for Naive Bayes
#	dummy - allocates labels with equal probability
# To use stopwords, supply 'stopwords' as the final argument.

pos_input = sys.argv[1]
neg_input = sys.argv[2]

vectorize_method = sys.argv[3]
classify_method = sys.argv[4]
stop_words = sys.argv[5]

# NOTE: This function, and only this function, borrowed from scikit-learn documentation,
# for purposes of having a prettier figure for the confusion matrix.
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
	# Read in files
	with open(pos_input, 'r') as pos_file:
		pos_data = pos_file.readlines()

	with open(neg_input, 'r') as neg_file:
		neg_data = neg_file.readlines()

	# Split into training and testing (90% training, 10% testing)
	pos_train = pos_data[:int(0.9*len(pos_data))]
	pos_test = pos_data[int(0.9*len(pos_data)):]
	neg_train = neg_data[:int(0.9*len(neg_data))]
	neg_test = neg_data[int(0.9*len(neg_data)):]

	train_data = pos_train + neg_train
	test_data = pos_test + neg_test

	# Set up class labels
	train_classes = []
	test_classes = []
	for x in range(len(pos_train)):
		train_classes.append("pos")
	for x in range(len(neg_train)):
		train_classes.append("neg")
	for x in range(len(pos_test)):
		test_classes.append("pos")
	for x in range(len(neg_test)):
		test_classes.append("neg")

	# Set up stop word parameters
	if stop_words == "stopwords":
		stop_words = set(stopwords.words("english"))	# Get stopwords from NLTK
	else:
		stop_words = None

	# Vectorize the data
	if vectorize_method == "unigram":
		vectorizer = CountVectorizer(ngram_range=(1,1), decode_error='replace', min_df=0.0001, stop_words=stop_words)
	elif vectorize_method == "bigram":
		vectorizer = CountVectorizer(ngram_range=(2,2), decode_error='replace', min_df=0.0001, stop_words=stop_words)
	elif vectorize_method == "both":
		vectorizer = CountVectorizer(ngram_range=(1,2), decode_error='replace', min_df=0.0001, stop_words=stop_words)

	train_vectors = vectorizer.fit_transform(train_data)
	test_vectors = vectorizer.transform(test_data)

	# Classify the data
	if classify_method == "logreg":
		classifier = linear_model.LogisticRegression()
	elif classify_method == "svm":
		classifier = svm.LinearSVC()
	elif classify_method == "nb":
		classifier = MultinomialNB(alpha=0.7)
	elif classify_method == "dummy":
		classifier = DummyClassifier(strategy="uniform")

	classifier.fit(train_vectors, train_classes)
	prediction = classifier.predict(test_vectors)

	# How good was it?
	total = len(test_classes)
	hits = 0
	for index, i in enumerate(test_classes):
		if test_classes[index] == prediction[index]:
			hits = hits + 1
	percentage_hits = float(hits)/float(total)
	print "MODEL: " + vectorize_method + " " + classify_method
	print "PERCENTAGE CORRECTLY CLASSIFIED: " + str(percentage_hits)

	# Print confusion matrix (optional section).
	cnf_matrix = confusion_matrix(test_classes, prediction)
	print cnf_matrix
	plot_confusion_matrix(cnf_matrix, classes=["pos", "neg"], normalize=False, title="Normalized confusion matrix")
	plt.figure()
	plt.show()
