"""
Implementation of the collapsed Gibbs sampler for Sentiment-LDA, described in
Sentiment Analysis with Global Topics and Local Dependency (Li, Huang and Zhu)
"""

import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize,sent_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
import dill
import os
st = PorterStemmer()


MAX_VOCAB_SIZE = 50000


def sample_from_dirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution

    """
    return np.random.dirichlet(alpha)


def sample_from_categorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()


def word_indices(word_occurance_vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in word_occurance_vec.nonzero()[0]:
        for i in range(int(word_occurance_vec[idx])):
            yield idx


class SentimentLDAGibbsSampler:

    def __init__(self, num_topics, alpha, beta, gamma, num_sentiments=2):
        """
        num_topics: Number of topics in the model
        num_sentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_topics = num_topics
        self.num_sentiments = num_sentiments

    def process_single_review(self, review, d=None):
        """
        Convert a raw review to a string of words
        """
        letters_only = re.sub("[^a-zA-Z]", " ", review)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [st.stem(w) for w in words if w not in stops]
        return(" ".join(meaningful_words))

    def proces_reviews(self, reviews, save_as=None, save_override=False):
        
        if not save_override and save_as and os.path.isfile(save_as):
            [word_occurence_matrix, self.vectorizer] = dill.load(open(save_as,'r'))
            return word_occurence_matrix
        processed_reviews = []
        i = 0
        for review in reviews:
            if((i + 1) % 1000 == 0):
                print ("Review %d of %d" % (i + 1, len(reviews)))
            processed_reviews.append(self.process_single_review(review, i))
            i += 1
        self.vectorizer = CountVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words="english",
                                          max_features=MAX_VOCAB_SIZE)
        train_data_features = self.vectorizer.fit_transform(processed_reviews)
        word_occurence_matrix = train_data_features.toarray()
        if save_as:
            dill.dump([word_occurence_matrix, self.vectorizer], open(save_as, 'w'))
        return word_occurence_matrix

    def _initialize_(self, reviews, save_as=None, save_override=False):
        """
        word_occurance_matrix: num_docs x vocab_size matrix encoding the
        bag of words representation of each document
        """
        self.word_occurance_matrix = self.proces_reviews(reviews, save_as, save_override)
        num_docs, vocab_size = self.word_occurance_matrix.shape

        # Pseudocounts
        self.n_dt = np.zeros((num_docs, self.num_topics))
        self.n_dts = np.zeros((num_docs, self.num_topics, self.num_sentiments))
        self.n_d = np.zeros((num_docs))
        self.n_vts = np.zeros((vocab_size, self.num_topics, self.num_sentiments))
        self.n_ts = np.zeros((self.num_topics, self.num_sentiments))
        self.topics = {}
        self.sentiments = {}
        self.prior_sentiment = {}

        alpha_vec = self.alpha * np.ones(self.num_topics)
        gamma_vec = self.gamma * np.ones(self.num_sentiments)

        for i, word in enumerate(self.vectorizer.get_feature_names()):
            synsets = swn.senti_synsets(word)
            pos_score = np.mean([s.pos_score() for s in synsets])
            neg_score = np.mean([s.neg_score() for s in synsets])
            if pos_score >= 0.1 and pos_score > neg_score:
                self.prior_sentiment[i] = 1
            elif neg_score >= 0.1 and neg_score > pos_score:
                self.prior_sentiment[i] = 0

        for d in range(num_docs):

            topic_distribution = sample_from_dirichlet(alpha_vec)
            sentiment_distribution = np.zeros(
                (self.num_topics, self.num_sentiments))
            for t in range(self.num_topics):
                sentiment_distribution[t, :] = sample_from_dirichlet(gamma_vec)
            for i, w in enumerate(word_indices(self.word_occurance_matrix[d, :])):
                t = sample_from_categorical(topic_distribution)
                s = sample_from_categorical(sentiment_distribution[t, :])

                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1

    def conditional_distribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (num_topics x num_sentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.num_topics, self.num_sentiments))
        first_factor = (self.n_dt[d] + self.alpha) / \
            (self.n_d[d] + self.num_topics * self.alpha)
        second_factor = (self.n_dts[d, :, :] + self.gamma) / \
            (self.n_dt[d, :] + self.num_sentiments * self.gamma)[:, np.newaxis]
        third_factor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= first_factor[:, np.newaxis]
        probabilities_ts *= second_factor * third_factor
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def get_top_k_words_by_likelihood(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(t, s | v) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (1, 2))
        pseudocounts /= normalizer[:, np.newaxis, np.newaxis]
        for t in range(self.num_topics):
            for s in range(self.num_sentiments):
                top_word_indices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                print (t, s, [vocab[i] for i in top_word_indices])

    def get_top_k_words(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        for t in range(self.num_topics):
            for s in range(self.num_sentiments):
                top_word_indices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                print (t, s, [vocab[i] for i in top_word_indices])

    def run(self, reviews, max_iters=30, save_as=None, save_override=False):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_(reviews, save_as, save_override)
        num_docs, vocab_size = self.word_occurance_matrix.shape
        for iteration in range(max_iters):
            print ("Starting iteration %d of %d" % (iteration + 1, max_iters))
            for d in range(num_docs):
                for i, v in enumerate(word_indices(self.word_occurance_matrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditional_distribution(d, v)
                    if v in self.prior_sentiment:
                        s = self.prior_sentiment[v]
                        t = sample_from_categorical(probabilities_ts[:, s])
                    else:
                        ind = sample_from_categorical(probabilities_ts.flatten())
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1