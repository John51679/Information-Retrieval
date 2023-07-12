# Information-Retrieval
This project was created as part of "Information Retrieval" subject in Computer Engineering &amp; Informatics Department (CEID) of University of Patras. It involves the retrieval process of movie data, using `elasticsearch` based on machine learning techniques. Four such techniques are implemented, in `Python` in file `AllTasks.py`, with the help of `elasticsearch`, `scikit-learn`, word_tokenize from `nltk.tokenize`, Word2Vec from `gensim.models` and one_hot and pad_sequences from `keras`.

- `Search1` function which uses elasticsearch's BM25 metric.
- `Search2` function which takes as extra parameter a user's ID and returns the relevant information as the sum of elasticsearch's BM25 metric, the user's personal rating on the movie (if it exists) and the average rating of that movie by all users (relevant csv file is `ratings.csv`.
- `Search3` function which takes each user and, for each movie category, groups him using K-means. Then for each group we fill missing ratings as the mean value of the group that the missing rating belongs to. Then use `Search2` to return results.
- `Search4` function which makes use of a neural network, after using Word2Vec to transform string data into vector space, for the purpose of predicting each user's missing movie rating. This function also allows the combination of it with the K-Means algorithm for maximum accuracy.
