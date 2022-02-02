import numpy as np
import sys
from tqdm import tqdm

sys.path.append("/home/kaliayev/Documents/ENSAE/elements_logiciels/word2vec_eltdm")

from word2vec_eltdm.common import (
    Tokenizer,
    VocabCreator,
    DataLoader,
    TokenCleaner,
    Preprocessor,
    Subsampler,
)
from word2vec_eltdm.word2vec_numpy import (
    NegWord2Vec,
    NegativeSamplingLoss,
    OptimizeNSL,
    train_NSL,
)


@profile
def train_wrapper(epochs, model, train_dataloader, criterion, optimizer, n_samples):
    for epoch in tqdm(range(epochs)):
        print(f"###################### EPOCH {epoch} ###########################")

        train_loss = train_NSL(model, train_dataloader, criterion, optimizer, n_samples)
        print("Training loss:", train_loss)

        # update learning rate
        optimizer.update_lr(epoch)


if __name__ == "__main__":
    datapath = "/home/kaliayev/Documents/ENSAE/elements_logiciels/word2vec_eltdm/data/text8.txt"

    RATIO = 0.1
    return_only_train = True
    tokenizer = Tokenizer(datapath)
    token_cleaner = TokenCleaner(freq_threshold=5)
    vocab_creator = VocabCreator()
    text8_dataset = Preprocessor(
        tokenizer, token_cleaner, vocab_creator, RATIO, return_only_train
    ).preprocess()

    print("Size of our vocabulary:", len(text8_dataset.tokens_to_id))
    print("Number of tokens in our train dataset:", len(text8_dataset.train_tokens))

    subsampler = Subsampler(text8_dataset.train_tokens)
    text8_dataset.train_tokens, text8_dataset.frequencies = subsampler.subsample()

    print(
        "Size of our vocabulary after subsampling of frequent words, for train:",
        len(text8_dataset.tokens_to_id),
    )
    print("Number of tokens in train dataset:", len(text8_dataset.train_tokens))

    window = 5
    batch_size = 256
    train_dataloader = DataLoader(
        text8_dataset, text8_dataset.train_tokens, window, batch_size
    )

    # defining the parameters
    len_vocab = len(text8_dataset.tokens_to_id)
    embedding_size = 300
    learning_rate = 1e-3
    n_samples = 5

    # hyperparameters for optimizer
    epochs = 1
    decay_rate = learning_rate / epochs
    method = "none"  # or "none", "exp_decay", "step_decay", "time_based"

    # Get our noise distribution
    word_freqs = np.array(sorted(text8_dataset.frequencies.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75))

    # instantiate the model
    model = NegWord2Vec(
        len_vocab, embedding_size, noise_dist=noise_dist, best_val_loss=0.0
    )
    model.initialize_weights()

    # using the loss that we defined
    criterion = NegativeSamplingLoss()
    optimizer = OptimizeNSL(model, learning_rate, decay_rate, method)

    # train for some number of epochs
    train_wrapper(epochs, model, train_dataloader, criterion, optimizer, n_samples)
