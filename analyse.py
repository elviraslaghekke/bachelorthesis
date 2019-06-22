import numpy as np

def get_best_and_worst(reviews, predictions, n=3):
    """Show the n reviews with highest and lowest predicted ratings."""

    p_index = np.argpartition(predictions, (0, n))[:n]
    print("Most positive reviews:")
    for i in p_index:
        print("rating:", predictions[i])
        print(reviews[i])

    n_index = np.argpartition(predictions, (-0, -n))[-n:]
    print("Most negative reviews:")
    for i in n_index:
        print("rating:", predictions[i])
        print(reviews[i])


def most_informative_features(model, vectorizer, n=20):
    """Show the most informative features from an sklearn Vectorizer."""

    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(model.coef_, feature_names))

    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
