import json
import time
import numpy as np
from analyse import most_informative_features, get_best_and_worst
from features import ReviewLengthInWords
from features import AverageWordLength
from features import TypeTokenRatio
from features import NumberOfAdjectives
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict

stemmer = SnowballStemmer("dutch")


def read_data(file_name, title_weight=1):
    """Read the data and return a list of review texts and a list of review ratings."""
    reviews = []
    ratings = []
    with open(file_name, encoding='utf8') as f:
        data = json.load(f)
        for review in data:
            title = review['review_title']
            body = review['review_text']
            text = (title + " ") * title_weight + body
            reviews.append(text)
            ratings.append(int(review['review_rating']))
        return reviews, ratings


def tokenize(text, method='nltk', stemming=True):
    """Tokenize text with three possible methods, and return a list of tokens."""
    if method == 'nltk':
        tokens = word_tokenize(text)  # keep punctuation
    elif method == 'sklearn':
        tokenizer = TfidfVectorizer().build_tokenizer()  # remove punctuation
        tokens = tokenizer(text)
    else:
        tokens = text.split()

    if stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def train(model, reviews, ratings):
    """Show training cross-validation results, train and return the model."""

    scores = cross_val_score(model, reviews, ratings, cv=10, scoring='neg_mean_squared_error')
    print("System mean MSE on training: ", abs(np.mean(scores)))

    start = time.time()  # counts running time
    model = model.fit(reviews, ratings)
    end = time.time()
    print("Training runtime:", (end - start), "seconds")  # prints total running time

    most_informative_features(model.named_steps['regression'], dict(model.named_steps['features'].transformer_list).get('unigrams_bigrams'))

    return model

def show_baselines(train_ratings, test_ratings):
    print("Baselines for train data:")
    baseline = [3] * len(train_ratings)
    mse = mean_squared_error(train_ratings, baseline)
    print("First baseline (predict middle rating (3) as every score) MSE: {}".format(str(mse)))

    mean_rating = np.mean(train_ratings)
    print(mean_rating)
    baseline = [mean_rating] * len(train_ratings)
    mse = mean_squared_error(train_ratings, baseline)
    print("Second baseline (predict the mean rating ({}) as every score) MSE: {}".format(str(mean_rating), str(mse)))
    print()

    print("Baselines for test data:")
    baseline = [3] * len(test_ratings)
    mse = mean_squared_error(test_ratings, baseline)
    print("First baseline (predict middle rating (3) as every score) MSE: {}".format(str(mse)))
    print()

    baseline = [mean_rating] * len(test_ratings)
    mse = mean_squared_error(test_ratings, baseline)
    print("Second baseline (predict the mean rating ({}) as every score) MSE: {}".format(str(mean_rating), str(mse)))
    print()

def test(model, reviews, ratings):
    """Test the trained model on new data and print the MSE."""
    predictions = model.predict(reviews)
    mse = mean_squared_error(ratings, predictions)
    print("System mean MSE on test: {}".format(str(mse)))
    get_best_and_worst(reviews, predictions)

def main():
    train_reviews, train_ratings = read_data('train_reviews.json', title_weight=3)
    test_reviews, test_ratings = read_data('test_reviews.json', title_weight=3)
#    nl_stop_words = get_stop_words('dutch')

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('unigrams_bigrams', TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, lowercase=True)),
            #('review_length_in_words', ReviewLengthInWords()),
            ('average_word_length', AverageWordLength())
            #('type_token_ratio', TypeTokenRatio()),
            #('number_of_adjectives', NumberOfAdjectives())
        ])),
        ('regression', linear_model.Ridge())
        ]
    )

    show_baselines(train_ratings, test_ratings)
    trained_model = train(pipeline, train_reviews, train_ratings)
    test(trained_model, test_reviews, test_ratings)

if __name__ == '__main__':
    main()
