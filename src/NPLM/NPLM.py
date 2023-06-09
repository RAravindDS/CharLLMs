import torch
import random 
from tqdm.auto import tqdm 
import torch.nn.functional as F 
import matplotlib.pyplot as plt

#A Neural Probabilistic Language Model
class NPLM: 

    def __init__(self, text_path, model_parameters): 
        self.text_path = text_path 
        self.model_parameters = model_parameters
        self.update_params()
        
    
    def update_params(self):
        block_size = self.model_parameters['block_size']
        train_size = self.model_parameters['train_size'] 
        epochs = self.model_parameters['epochs']
        batch_size = self.model_parameters['batch_size']
        hidden_layer = self.model_parameters['hidden_layer']
        embedding_dimension = self.model_parameters['embedding_dimension']
        learning_rate = self.model_parameters['learning_rate']
        
        self._load_prepare(block_size, train_size)
        self.configure_model_dimensions(epochs, batch_size, hidden_layer, embedding_dimension, learning_rate)


    def _load_prepare(self, block_size:int=3, train_size=0.8): 
        data = open(self.text_path, "r").read().splitlines()
        self.chars = sorted(list(set("".join(data))))

        print(f"Len of words: {len(data)}") 
        print(f"Len of unique characters in dataset: {len(self.chars)}") 
        
        stoi = { v:k+1 for k, v in enumerate(self.chars)}
        stoi['.'] = 0
        itos = { v:k for k, v in stoi.items()}
        
        self.stoi = stoi 
        self.itos = itos 
        self.block_size = block_size 

        def build_dataset(words): 
            block_size = self.block_size 
            X, Y = [], [] 
            for w in words: 
                context = [0] * block_size  
                for ch in w + ".": 
                    ix = self.stoi[ch]
                    X.append(context)
                    Y.append(ix)
                    context = context[1:] + [ix]   

            X = torch.tensor(X)
            Y = torch.tensor(Y)
            
            return X, Y 
        
        n2_slize=train_size + 0.1
        n1 = int(len(data)*train_size)
        n2 = int(len(data)*n2_slize)
        random.shuffle(data)

        self.train_X, self.train_Y = build_dataset(data[:n1]) 
        self.val_X, self.val_Y = build_dataset(data[n1:n2])
        self.test_X, self.test_Y = build_dataset(data[n2:])
    
        print()
        print(f"Block size: {self.block_size}")
        print(f"Len of TrainX: {len(self.train_X)}, Len of TrainY: {len(self.train_Y)}")
        print(f"Len of ValX: {len(self.val_X)}, Len of ValY: {len(self.val_Y)}")
        print(f"Len of TestX: {len(self.test_X)}, Len of TestY: {len(self.test_Y)}")
        
        
    def configure_model_dimensions(self, epochs=1000, batch_size=32, hidden_layer=100, embedding_dimension=2, learning_rate=0.1):

        g = torch.Generator().manual_seed(2147483647)
        self.input_layer_size = embedding_dimension * self.block_size
        
        self.b1 = torch.randn(hidden_layer, generator=g)
        self.W1 = torch.randn( (self.input_layer_size, hidden_layer), generator=g)
        self.b2 = torch.randn( len(self.chars)+1, generator=g)
        self.W2 = torch.randn( (hidden_layer, 27 ), generator=g) 
        self.C = torch.randn( (len(self.chars)+1, embedding_dimension) )  

        parameters = [self.C, self.b1, self.b2, self.W1, self.W2] 

        self.batch_size = batch_size 
        self.epochs = epochs 
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.hidden_layer = hidden_layer
        self.g = g

        print(f"Total of paramters: {sum(p.nelement() for p in parameters)}")

        ## making all require grads = True 
        for p in parameters:
            p.requires_grad=True 
            
            
    def train_model(self): 
        
        lossi = []
        for br in tqdm(range(self.epochs)): 
            
            ## forward pass
            ix = torch.randint(0, self.train_X.shape[0], (self.batch_size,) )
            emb = self.C[self.train_X[ix]]  # mini batch
            h = torch.tanh( emb.view(-1, self.input_layer_size)@self.W1 + self.b1) 
            logits = h @ self.W2 + self.b2 
            loss = F.cross_entropy(logits, self.train_Y[ix]) 
            
            lossi.append(loss.item())

            ## backward pass 
            for p in self.parameters: 
                p.grad = None 
            loss.backward()

            ## update 
            lri =  self.learning_rate if br < 8000 else 0.001
            for p in self.parameters: 
                p.data += -lri * p.grad 
            
            if br % 100 == 0: 
                print(f"Training Loss ({br}) {loss.item()}")
                
        # all loss 
        ix = torch.randint(0, self.val_X.shape[0], (self.batch_size,) )
        emb = self.C[self.val_X[ix]]  # evaluvating with dev set 
        h = torch.tanh( emb.view(-1, self.input_layer_size)@self.W1 + self.b1) 
        logits = h @ self.W2 + self.b2 
        loss = F.cross_entropy(logits, self.val_Y[ix])
        print(f"Validation loss: {loss}") 
        
        plt.plot(lossi, 'g')
        plt.xlabel("Epochs")
        plt.ylabel("LearningRate")
        plt.title("Learnign rate over epochs")
        min_index =  lossi.index(min(lossi))
        plt.plot( min_index, lossi[min_index], 'ro' )
        
        return plt.show()
    
    
    def sampling(self, words_needed=10): 
        
        all_words=[]
        for _ in range(words_needed):
            
            out = []
            context = [0] * self.block_size 
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                
                ix = torch.multinomial(probs, num_samples=1, generator=self.g).item()
                context = context[1:] + [ix]
                out.append(ix)
                
                if ix == 0:
                    break
                
            print(''.join(self.itos[i] for i in out))
            all_words.append("".join(self.itos[i] for i in out))