import numpy as np

features = np.load('./features/bert_movie-corpus_text_features.npy')

train_idx = np.load('./metadata/movie/train_idx.npy')
learn_idx = np.load('./metadata/movie/learn_idx.npy')
test_idx = np.load('./metadata/movie/test_idx.npy')

print(train_idx[:10])
print(learn_idx[:10])
print(test_idx[:10])

train_features = features[train_idx]
learn_features = features[learn_idx]
test_features = features[test_idx]

np.save('./features/bert_movie-corpus_text_features_train.npy', train_features)
np.save('./features/bert_movie-corpus_text_features_learn.npy', learn_features)
np.save('./features/bert_movie-corpus_text_features_test.npy', test_features)
