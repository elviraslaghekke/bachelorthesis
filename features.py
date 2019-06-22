from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import alpino as alp
from nltk.tag import UnigramTagger
import re


class ReviewLengthInWords(BaseEstimator, TransformerMixin):
    """Calculate and return the length of reviews in words."""
    def fit(self, reviews, y=None):
        return self

    def transform(self, reviews, y=None):
        lengths = []
        for review in reviews:
            length = len(review.split())
            lengths.append([length])
        return lengths


class AverageWordLength(BaseEstimator, TransformerMixin):
    """Calculate and return the average length of words in the reviews."""
    def fit(self, reviews, y=None):
        return self

    def transform(self, reviews, y=None):
        word_lengths = []
        for review in reviews:
            wl = 0
            w = 0
            for word in review.split():
                wl += len(word)
                w += 1
            word_lengths.append([wl/w])
        return word_lengths


class TypeTokenRatio(BaseEstimator, TransformerMixin):
    """Calculate and return the type token ratio (no. of types/no. of tokens)."""
    def fit(self, reviews, y=None):
        return self

    def transform(self, reviews, y=None):
        type_token_ratio = []
        for review in reviews:
            tokens = re.findall(r"[\w']+|[.,!?;]", review)
            tokens = [token.lower() for token in tokens]
            types = set(tokens)
            ratio = len(types) / len(tokens)
            type_token_ratio.append([ratio])
        return type_token_ratio


class NumberOfAdjectives(BaseEstimator, TransformerMixin):
    """Count and return the number of adjectives in the reviews."""
    def fit(self, reviews, y=None):
        return self

    def transform(self, reviews, y=None):
        number_of_adjectives = []
        training_corpus = alp.tagged_sents()
        unitagger = UnigramTagger(training_corpus)
        pos_tag = unitagger.tag
        for review in reviews:
            tokens = re.findall(r"[\w']+|[.,!?;]", review)
            adj = 0
            for token in pos_tag(tokens):
                if token[1] == 'adj':
                    adj += 1
            number_of_adjectives.append([adj])
        return number_of_adjectives
