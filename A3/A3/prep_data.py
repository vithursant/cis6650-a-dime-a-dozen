""" Supporting code for ENGG*6500 Assignment 3
Partly Sunny with a Chance of Hashtags
https://www.kaggle.com/c/crowdflower-weather-twitter
"""
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse as sp

import pandas as pd
from gensim.models import word2vec
import logging

import array
#import _pickle as cPickle
import cPickle as pickle
import os

logger = logging.getLogger(__name__)

data_path = os.getcwd() #os.path.join('/scratch', os.environ['USER'])

def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))

def tokenize(raw_documents):
    """ Note that tokenization and counting are coupled in sklearn
    This hijacks the tokenization from CountVectorizer._count_vocab
    """

    vocabulary = defaultdict(None)
    vocabulary.default_factory = vocabulary.__len__

    analyze = vectorizer.build_analyzer()


    j_indices = _make_int_array()
    indptr = _make_int_array()
    indptr.append(0)

    for doc in raw_documents:
        for feature in analyze(doc):
            try:
                j_indices.append(vocabulary[feature])
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue
        indptr.append(len(j_indices))

    # disable defaultdict behaviour
    vocabulary = dict(vocabulary)
    if not vocabulary:
        raise ValueError("empty vocabulary; perhaps the documents only"
                         " contain stop words")

    # some Python/Scipy versions won't accept an array.array:
    if j_indices:
        j_indices = np.frombuffer(j_indices, dtype=np.intc)
    else:
        j_indices = np.array([], dtype=np.int32)
    indptr = np.frombuffer(indptr, dtype=np.intc)
    values = np.ones(len(j_indices))

    return j_indices, indptr, vocabulary


def make_list(raw_documents):
    """ takes array of strings (documents)
    return list of lists
    where each element is a preprocessed sentence (list of words)
    """

    vectorizer = CountVectorizer(stop_words=None)

    # note that, even with stop_words=None
    # the tokenizer gets rid of anything less than 2 characters
    # so punctuation is removed, as well as 1-letter words
    analyze = vectorizer.build_analyzer()

    x = []

    for doc in raw_documents:
        x.append(analyze(doc))

    return x


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    word2vec_model_file = "word2vec.model"
    data_file = os.getcwd() +  "/preprocessed_data.pkl"#os.path.join(data_path, "preprocessed_data.pkl")
    n_features = 20  # feature vector length
    window = 5  # how far to look back when training the word2vec model
    min_count = 5  # ignore all words with a total frequency lower than this

    # read training data, test_data from csv
    train_data = pd.read_table(os.path.join(data_path, 'train.csv'),
                               index_col=0, sep=',', header=0)
    test_data = pd.read_table(os.path.join(data_path, 'test.csv'),
                              index_col=0, sep=',', header=0)

    # ss = ['k%d' % (i + 1) for i in xrange(5)]
    # ws = ['k%d' % (i + 1) for i in xrange(4)]
    # ks = ['k%d' % (i + 1) for i in xrange(15)]

    # convert tweets into a list of lists
    # each element represents a tweet
    # the tweet is a list of words
    tweets_train = make_list(train_data['tweet'])
    tweets_test = make_list(test_data['tweet'])

    # train word2vec on both training and test data
    # note this is allowed by the competition (i.e. transductive learning)
    # we may not want to do this but it avoids the problem of missing rep for
    # words in the test data  that don't show up in the training data
    #
    # we would actually be better off learning this model on a larger dataset
    model = word2vec.Word2Vec(tweets_train + tweets_test,
                              size=n_features, window=window,
                              min_count=min_count)
    #model.save(word2vec_model_file)
    #model = word2vec.Word2Vec.load('word2vec.model')

    # form the data out of the word vectors
    # each tweet is represented as a n_words x n_features array
    train_X = []
    test_X = []

    m = model.wv.syn0.mean(axis=0)  # mean vector

    def get_features(x):
        try:
            return model[x]
        except KeyError:
            # no matching vector, replace with mean vector
            return m

    logger.info('Mapping training set to word vectors')
    for i, tweet in enumerate(tweets_train):
        features = map(get_features, tweet)
        train_X.append(np.asarray(features))

    train_y = train_data.ix[:,3:]
    
    logger.info('Mapping test set to word vectors')
    for tweet in tweets_test:
        features = map(get_features, tweet)
        test_X.append(np.asarray(features))

    # split off some data for validation
    valid_X = train_X[50000:]
    valid_y = train_y[50000:]
    train_X = train_X[:50000]
    train_y = train_y[:50000]

    # no labels for test data
    data = (
        (train_X, train_y),
        (valid_X, valid_y),
        (test_X,),
    )

    print ("...saving data pickle: %s" % data_file)
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
