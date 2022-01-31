import numpy as np
import sys
from tqdm import tqdm

import torch
import torch.optim as optim

sys.path.append('/home/kaliayev/Documents/ENSAE/elements_logiciels/word2vec_eltdm')

from word2vec_eltdm.common import Tokenizer, VocabCreator, DataLoader, TokenCleaner, Preprocessor, Subsampler
from word2vec_eltdm.word2vec_accelerated import PytorchNegWord2Vec, NegativeSamplingLoss, train_NSL


@profile
def train_wrapper(epochs, model, train_dataloader, criterion, optimizer, n_samples, device):
    for epoch in tqdm(range(epochs)):
        print(f"###################### EPOCH {epoch} ###########################")
        train_loss = train_NSL(model, train_dataloader, criterion, optimizer, n_samples, device)
        print("Training loss:", train_loss.item())

if __name__=="__main__":
    datapath = "/home/kaliayev/Documents/ENSAE/elements_logiciels/word2vec_eltdm/data/text8.txt"

    RATIO = 0.1
    return_only_train = True
    tokenizer = Tokenizer(datapath)
    token_cleaner = TokenCleaner(freq_threshold=5)
    vocab_creator = VocabCreator()
    text8_dataset = Preprocessor(tokenizer, token_cleaner, vocab_creator, RATIO, return_only_train).preprocess()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Size of our vocabulary:", len(text8_dataset.tokens_to_id))
    print("Number of tokens in our train dataset:", len(text8_dataset.train_tokens))

    subsampler = Subsampler(text8_dataset.train_tokens)
    text8_dataset.train_tokens, text8_dataset.frequencies = subsampler.subsample()

    print("Size of our vocabulary after subsampling of frequent words, for train:", len(text8_dataset.tokens_to_id))

    window = 5
    batch_size = 256
    train_dataloader = DataLoader(text8_dataset, text8_dataset.train_tokens, window, batch_size)

    # defining the parameters
    len_vocab = len(text8_dataset.tokens_to_id)
    embedding_size = 300
    learning_rate = 1e-3
    n_samples = 5
    epochs = 1

    # Get our noise distribution
    word_freqs = np.array(sorted(text8_dataset.frequencies.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75))

    # instantiate the model
    model = PytorchNegWord2Vec(
        len_vocab,
        embedding_size,
        noise_dist=noise_dist,
        device=device
    ).to(device)
    model.initialize_weights()

    criterion = NegativeSamplingLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # train for some number of epochs
    train_wrapper(epochs, model, train_dataloader, criterion, optimizer, n_samples, device)
