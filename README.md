# Hybrid Query Benchmark




This repository is the benchmark for hybrid query of boolean filter and vector similarity search.

There are two datasets, the single table dataset is based on the Danish Fungi Dataset [1], and the multi table dataset is based on the Cornell Movie-Dialogs Corpus [2]. For more information about the dataset, please refer to the [homepage](https://hybridqueriesbenchmark.github.io/).


## Load dataset and do queries

Scripts in `experiments` show you how to load data of the benchmark into database and do queries. You need to install DBMS server in your machine, download and unzip the benchmark, and modify the `path`, `host` and `port` in the scripts to do queries.


## Disclaimer

The original license of Danish Fungi Dataset is [here](LICENSE-fungi) and the original license of Cornell Movie-Dialogs Corpus is [here](LICENSE-movie), please comply with the original open source license of the dataset.



## References

1. Picek L, Šulc M, Matas J, et al. Danish fungi 2020-not just another image recognition dataset[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2022: 1525-1535.
2. Danescu-Niculescu-Mizil C, Lee L. Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs[J]. arXiv preprint arXiv:1106.3077, 2011.
