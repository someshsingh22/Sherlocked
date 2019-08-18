
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
import re
from argparse import Namespace
import time
from tqdm import tqdm_notebook, tnrange
import seaborn as sns
from torch.optim.lr_scheduler import StepLR


# In[2]:


def calculate_time(function):
    def inner(*args,**kwargs):
        begin=time.time()
        function(*args,**kwargs)
        end=time.time()
        print("This took {}s".format(end-begin))
    return inner

class Author:
    def __init__(self,flags):
        self.flags=flags
        self.Data=Data(flags)
        self.Brain=Brain(flags,self.Data.n_vocab)
        self.Pen=Pen(flags,self.Data.vocab_to_int,self.Data.int_to_vocab)
    def Train(self,epochs):
        self.Brain.Train(epochs,self.Data,self.Pen)
        

class Pen:
    def __init__(self,flags,vocab_to_int,int_to_vocab):
        self.flags=flags
        self.vocab_to_int,self.int_to_vocab=vocab_to_int,int_to_vocab
    
    def write(self,Brain,words="Harry is",temperature=0.5,length=70):
        probs=[80-60*temperature,60-20*temperature,50,40+20*temperature,20+60*temperature]
        probs=[prob/250 for prob in probs]

        words=words.split()
        Brain.eval()
        state_h, state_c = Brain.zero_state(1)
        state_h = state_h.to(self.flags.device)
        state_c = state_c.to(self.flags.device)

        for w in words:
            ix = torch.tensor([[self.vocab_to_int[w]]]).to(self.flags.device)
            output, (state_h, state_c) = Brain(ix, (state_h, state_c))
            top_k=int(np.random.choice(np.arange(1,6),1,p=probs))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(self.int_to_vocab[choice])

        for _ in range(length):
            ix = torch.tensor([[choice]]).to(self.flags.device)
            output, (state_h, state_c) = Brain(ix, (state_h, state_c))
            top_k=int(np.random.choice(np.arange(1,6),1,p=probs))

            _, top_ix = torch.topk(output[0], k=top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])
            words.append(self.int_to_vocab[choice])
    
        prediction=' '.join(words)
        print(prediction)
        return prediction

# Data Class
class Data :
    def __init__(self,flags):
        self.flags=flags
    
        # Pre Process Data
        self.pre_process()
    
  # Prepare Data For Training
    @calculate_time
    def pre_process(self):
        start=time.time()
        # Read datat and split to words from files
        text=open(self.flags.data_dir).read().split()
        print("Dataset is of {} words".format(len(text)))

        # Create Frequency Dictionary
        word_counts = Counter(text)
        self.sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

        # Word2Vec Mapping
        self.int_to_vocab = {k: w for k, w in enumerate(self.sorted_vocab)}
        self.vocab_to_int = {w: k for k, w in self.int_to_vocab.items()}
        self.n_vocab = len(self.int_to_vocab)

        # Create Input-Output Data
        int_text = [self.vocab_to_int[w] for w in text]
        num_batches = int(len(int_text) / (self.flags.seq_size * self.flags.batch_size))
        in_text = int_text[:num_batches * self.flags.batch_size * self.flags.seq_size]
        out_text = np.zeros_like(in_text)
        out_text[:-1] = in_text[1:]
        out_text[-1] = in_text[0]
        in_text = np.reshape(in_text, (self.flags.batch_size, -1))
        out_text = np.reshape(out_text, (self.flags.batch_size, -1))
        print("Data Preprocessing complete with {} words".format(self.n_vocab))
        self.in_text=in_text
        self.out_text=out_text

  # Create Input-Output Batch Generator 
    def get_batches(self):
        num_batches = np.prod(self.in_text.shape) // (self.flags.seq_size * self.flags.batch_size)
        for i in range(0, num_batches * self.flags.seq_size, self.flags.seq_size):
            yield self.in_text[:, i:i+self.flags.seq_size], self.out_text[:, i:i+self.flags.seq_size]

class Brain(nn.Module):
    def __init__(self,flags,n_vocab,):
        super(Brain,self).__init__()
        self.flags=flags
        self.embedding= nn.Embedding(n_vocab, flags.embedding_size)
        self.lstm=nn.LSTM(flags.embedding_size, flags.lstm_size, batch_first=True, num_layers=flags.num_layers, dropout=flags.dropout)
        self.dense = nn.Linear(flags.lstm_size, n_vocab)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.flags.lr)
        self.loss_value=0
        self.to(flags.device)
        self.plot_loss=[]
        # gamma = decaying factor
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.96)
    
  # Forward Pass
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state

  # ZeroState Init
    def zero_state(self,batch_size):
        return (torch.zeros(self.flags.num_layers, batch_size, self.flags.lstm_size),
                torch.zeros(self.flags.num_layers, batch_size, self.flags.lstm_size))
  
  #Train Function
    @calculate_time
    def Train(self,epochs,Data,Pen):
        self.train()
        for epoch in tnrange(1,epochs+1,desc="Epoch Loop"):
            start=time.time()
            batches=Data.get_batches()
            state_h, state_c = self.zero_state(self.flags.batch_size)

            # Transfer data to GPU
            state_h = state_h.to(self.flags.device)
            state_c = state_c.to(self.flags.device)
            
            for x, y in (batches):
            #for x, y in (batches,desc="Batch Loop",total=87,leave=False):

                # Reset all gradients
                self.optimizer.zero_grad()

                # Transfer data to GPU
                x = torch.tensor(x).to(self.flags.device)
                y = torch.tensor(y).to(self.flags.device)

                logits, (state_h, state_c) = self(x, (state_h, state_c))
                self.loss = self.criterion(logits.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                self.loss_value = self.loss.item()

                # Update the network's parameters
                self.optimizer.step()
                self.loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(self.parameters(), self.flags.gradients_norm)
                #self.Optimizer.decay(self.loss)

                self.optimizer.step()
            callback='Loss: {} Time Taken : {}'.format(self.loss_value,time.time()-start)
            self.scheduler.step()
            print(callback)
            self.plot_loss.append(self.loss_value)
            if self.loss_value < 0.75 :
                break
            if epoch%10==0:
                print("-"*20,"\nEpoch {}:".format(epoch))
                Pen.write(self)
                print("-"*20+"\n")
                self.train()


# In[3]:


outfile=""
def prewrite(outfile,flags):
    outfile+=("-"*20+"\n")
    for key in flags.__dict__:
        val=flags.__dict__[key]
        if type(val) == int:
            outfile+=(str(key)+"\n")
            outfile+=(str(val)+"\n")
    outfile+=("-"*20+"\n")
    return outfile


# In[4]:


def commit_file(outfile,flags,epochs=3):
    print(prewrite(outfile,flags))
    Rowling=Author(flags)
    Rowling.Train(epochs)
    x=[i+1 for i in range(len(Rowling.Brain.plot_loss))]
    sns.lineplot(x,Rowling.Brain.plot_loss)


# In[7]:


read_path="../input/repository/someshsingh22-Sherlocked-39b34b9/Dataset/Clean/{}.txt"
write_path="{}_Results.txt"
file="HP_Stitched"
flags = Namespace(
    data_dir=read_path.format(file),
    batch_size=256,
    seq_size=64,
    embedding_size=256,
    num_layers=2,
    is_bidirectional=True,
    lstm_size=512,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    write_dir=write_path.format(file),
    lr=0.001,
    dropout=0.2,
    gradients_norm=5,
)


# In[6]:


flags.is_bidirectional=True
flags.lstm_size=512
flags.num_layers=2
commit_file(outfile,flags,100)


# In[ ]:


flags.is_bidirectional=True
flags.lstm_size=1024
flags.num_layers=1
commit_file(outfile,flags,100)


# In[ ]:


flags.is_bidirectional=False
flags.lstm_size=512
flags.num_layers=2
commit_file(outfile,flags,100)


# In[ ]:


flags.is_bidirectional=False
flags.lstm_size=1024
flags.num_layers=1
commit_file(outfile,flags,100)

