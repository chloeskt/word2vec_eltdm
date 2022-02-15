
# Parallelizing Word2Vec

This project is the done during the course "Éléments logiciels pour le traitement des données massives" given by [Xavier Dupré](http://www.xavierdupre.fr/)
and [Matthieu Durut](https://www.linkedin.com/in/matthieudurut) at ENSAE ParisTech. 

Final grade: 18/20

## About this project

The idea is to implement a naïve version (using continuous Skip-gram) of the [Word2Vec algorithm](https://arxiv.org/abs/1301.3781) developed by Mikolov et al.
This will be done in Python, using Numpy library. And compare this naïve version to a faster and more scalable version, inspired 
by the work of Ji & al, in their paper ["Parallelizing Word2Vec in Shared and Distributed Memory"](https://arxiv.org/abs/1604.04661).
Our implementation will be based on Pytorch and we shall compare the performances of the two algorithms in terms of training speed, 
inference time, parallel schemes, number of threads used, etc. 

All evaluations will be run on the same machine with 8 cores/16 threads (Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz), 32GB of RAM and one 
NVIDIA GeForce RTX 3080 (10GB VRAM).

## Installation 

The code was run and tested in Python 3.8.

````python
pip install -r requirements.txt
````

This will install Pytorch CPU version. To get the CUDA available version, you must follow the official documentation which can
be found [here](https://pytorch.org/get-started/locally/) to install the compatible version with your hardware.

To download files relevant for our word2vec.
```
python3 -m nltk.downloader stopwords
```

## Data

For this project, we used the same data as in ["Parallelizing Word2Vec in Shared and Distributed Memory"](https://arxiv.org/abs/1604.04661).
One can retrieve them by running:
```bash
mkdir data && cd data
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f
```
and
```bash
cd data
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
cat 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 > 1b
cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* >> 1b
rm -rf 1-billion-word-language-modeling-benchmark-r13output
```

The previous code has been taken from Ji et al. work and can be found in the corresponding [repository](https://github.com/IntelLabs/pWord2Vec/tree/master/data).

## About the structure of the project

#### `word2vec_eltdm` folder:

It contains all the source code divided in 3 main sub-folders:
- ``common``: common source code between the Numpy implementation and the Pytorch one.
- ``word2vec_accelerated``: Pytorch version of the models
- ``word2vec_numpy``: Numpy version of the models

#### ``notebooks`` folder:

Contains some notebooks to see model training and evaluation. 

#### ``speed_tests`` folder:

Contains the source code related to training speed evaluation. The sub-folder ``results`` contains the results of these
training speed tests.

## References

["Parallelizing Word2Vec in Shared and Distributed Memory"](https://arxiv.org/abs/1604.04661) by Ji. et al.

Their code has be found [here](https://github.com/IntelLabs/pWord2Vec).

[Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) by Mikolov et al. 

[Efficient Estimation of Words Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Mikolov et al. 
