# Beyond the stars

This is the repository for my bachelor thesis: Beyond the stars, Predicting ratings of Dutch restaurant reviews

###Abstract

For my bachelor thesis, I built a system that predicts the star rating of Dutch restaurant reviews. I used a data set containing 50.000 reviews from TripAdvisor.nl. I treated it as a supervised regression problem and used the Ridge model from sci-kit learn. I obtained the best results with lowercasing and stemming as processing methods. For features, I used unigrams, bigrams and the average word length of a review. Also, giving more weight to the review title especially improved the score. I used Mean Squared Error (MSE) to evaluate the system. I obtained a final score of 0.53 MSE. The system performed really well, considering the two baselines of 2.17 and 1.23 MSE. 