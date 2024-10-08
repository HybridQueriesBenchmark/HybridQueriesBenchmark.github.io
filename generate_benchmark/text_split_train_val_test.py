import json
import numpy as np
import random
import os
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

total = 304713
train_size = 284713
learn_size = 10000
test_size = 10000
assert(total == train_size + learn_size + test_size)

'''
    @brief: Split the dataset into train, learn and test sets
    @param: file_path: the path of the dataset
    @param: train_size: the size of the train set
    @param: learn_size: the size of the learn set
    @param: test_size: the size of the test set

    @return: 3 list, each list contains the id of the samples in the corresponding set
'''
def split_train_val_test(file_path, train_size, learn_size, test_size):
    total_idx = set([i for i in range(total)])
    learn = random.sample(total_idx, learn_size)
    learn = set(learn)
    test = random.sample([i for i in total_idx if i not in learn], test_size)
    test = set(test)
    train = [i for i in total_idx if i not in learn and i not in test]
    learn = list(learn)
    test = list(test)
    learn.sort()
    test.sort()
    return train, learn, test


if __name__ == "__main__":
    seed_everything(42)

    file_path = r"F:\datasets\NLP\movie-corpus\utterances.jsonl"

    train_idx, learn_idx, test_idx = split_train_val_test(file_path, train_size, learn_size, test_size)
    print(len(train_idx), len(learn_idx), len(test_idx))
    print(train_idx[:10], learn_idx[:10], test_idx[:10])

    np.save(r"./metadata/movie/train_idx.npy", np.array(train_idx))
    np.save(r"./metadata/movie/learn_idx.npy", np.array(learn_idx))
    np.save(r"./metadata/movie/test_idx.npy", np.array(test_idx))

    
