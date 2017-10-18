import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt

with open('kjv_preproc.txt', 'r') as kjv:
    corpus_raw = kjv.read()

# Map each unique word to an integer (and vice versa)
words = []
for word in corpus_raw.split():
    if word not in ".,:()\'-;—’‘": # Strip out the punctuation
        words.append(word)
words = set(words) # Remove duplicates
word2int = {}
int2word = {}
vocab_size = len(words)
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# Get a list of sentences
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    clean_sentence = []
    for item in sentence.split():
        if item not in ".,:()\'-;—’‘":
            clean_sentence.append(item)
    sentences.append(clean_sentence)

# Generate training data
data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

# Convert these numbers to one-hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size) # Fill with zeros...
    temp[data_point_index] = 1  # Except at the desired index
    return temp
x_train = [] # input word
y_train = [] # output word
for data_window in data:
    x_train.append(to_one_hot(word2int[ data_window[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_window[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


# Placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

# Training data -> embedded representation
EMBEDDING_DIM = 5
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) # Bias term
hidden_representation = tf.add(tf.matmul(x,W1), b1) # x*W1 + b1

# Embedded representation -> probability prediction about neighbor
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2)) # x*W2 + b2, softmax'ed to one-hot

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# Loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# Training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000
# Training loop:
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print(_, 'loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

# We had added a bias term, so we add it here
vectors = sess.run(W1 + b1)

def find_closest(word_index, vectors):
    min_dist = 100000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if np.linalg.norm(vector - query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = np.linalg.norm(vector - query_vector)
            min_index = index
    return min_index

# Examples of closest word to given word
print('john:', int2word[find_closest(word2int['john'], vectors)])
print('christ:', int2word[find_closest(word2int['christ'], vectors)])
print('lamb:', int2word[find_closest(word2int['lamb'], vectors)])
print('was:', int2word[find_closest(word2int['was'], vectors)])
print('word:', int2word[find_closest(word2int['word'], vectors)])
print('witness:', int2word[find_closest(word2int['witness'], vectors)])
print('light:', int2word[find_closest(word2int['light'], vectors)])
print('follow:', int2word[find_closest(word2int['follow'], vectors)])
print('testify:', int2word[find_closest(word2int['testify'], vectors)])
print('sin:', int2word[find_closest(word2int['sin'], vectors)])
print('grace:', int2word[find_closest(word2int['grace'], vectors)])
print('father:', int2word[find_closest(word2int['father'], vectors)])
print('disciples:', int2word[find_closest(word2int['disciples'], vectors)])
print('flesh:', int2word[find_closest(word2int['flesh'], vectors)])
print('son:', int2word[find_closest(word2int['son'], vectors)])
print('right:', int2word[find_closest(word2int['right'], vectors)])

# Reduce dimensions from 5 to 2 so we can see it on a graph
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)

# Normalize the data
normalizer = preprocessing.Normalizer()
vectors2 =  normalizer.fit_transform(vectors, 'l2')

# Plot it
fig, ax = plt.subplots(figsize=(20, 10))
for word in words:
    print(word, vectors2[word2int[word]][1])
    ax.annotate(word, (vectors2[word2int[word]][0],vectors2[word2int[word]][1] ))
plt.show()
