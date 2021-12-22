from word2vec_eltdm.word2vec_numpy import Tokenizer, VocabCreator, DataLoader
from word2vec_eltdm.word2vec_numpy.dataset import Dataset
from word2vec_eltdm.word2vec_numpy.token_cleaner import TokenCleaner
from word2vec_eltdm.word2vec_pytorch import PytorchWord2Vec

class Trainer:
    def __init__(
        self, 
        datapath: str, 
        ratio: float,
        hidden_size: int = 500,
        embedding_size: int = 300,
        window: int = 5,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        learning_rate: float = 1e-3,
        n_epochs: int = 3,
    )

    self.n_epochs = n_epochs
    # Prepare Dataset
    self.dataset = Preprocessor(
            Tokenizer(datapath),
            TokenCleaner(),
            VocabCreator(),
            ratio,
            ).preprocess()
    subsampler = Subsampler(self.dataset.train_tokens)
    self.dataset.train_tokens, self.dataset.frequencies = subsampler.subsample()

    self.init_dataloader(window, batch_size)

    self.model = PytorchWord2Vec(
        len(self.dataset.tokens_to_id),
        hidden_size,
        embedding_size,
    )
    model.initialize_weights()
    
    self.optimizer = torch.optim.SparseAdam(self.model.parameters(), learning_rate)
    self.criterion = PytorchNSL()
    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealignLR(
    #     self.optimizer, 
    #     len(self.train_dataloader)
    # )

    def init_dataloader(self, window, batch_size):
        self.train_dataloader = DataLoader(
            self.dataset, 
            self.dataset.train_tokens,
            window, 
            batch_size,
        )
        self.val_dataloader = DataLoader(
            self.dataset, 
            self.dataset.val_tokens,
            window, 
            batch_size,
        )
        self.test_dataloader = DataLoader(
            self.dataset, 
            self.dataset.test_tokens,
            window, 
            batch_size,
        )
        
    def train(self):
        for e in range(self.n_epochs):
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                steps += 1
                X, y = torch.Tensor(batch['X']), torch.Tensor(batch['y'])
                #X, y = X.to(device), y.to(device)
                X_vect, y_vect = self.model.forward(X, y)
                noise_vect = model.forward_noise(X.shape[0], 5)
                loss = criterion(X_vect, y_vect, noise_vect)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps % print_every == 0:
                    self.verbose(e, loss)

    def verbose(self, epoch, loss):
        print("Epoch: {}/{}".format(e+1, epochs))
        print("Loss: ", loss.item()) # avg batch loss at this point in training
        valid_examples, valid_similarities = cosine_similarity(self.model.W1)#, device=device)
        _, closest_idxs = valid_similarities.topk(6)

        valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
        for ii, valid_idx in enumerate(valid_examples):
            closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
            print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
        print("...\n")


class PytorchNSL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        #debugging
        #print(type(noise_vectors)) #it is a tensor
        
        #'neg' returns the negative of a tensor
        #print(noise_vectors)
        #print(noise_vectors.neg())
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
