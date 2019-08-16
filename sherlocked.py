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
from tqdm import tqdm

read_path="./Sherlocked/Dataset/Clean/{}.txt"
write_path="./Sherlocked/{}.txt"
file="cano"
flags = Namespace(
    data_dir=read_path.format(file),
    batch_size=256,
    seq_size=64,
    embedding_size=256,
    num_layers=1,
    is_bidirectional=False,
    lstm_size=512,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    write_dir=write_path.format(file),
    lr=0.01,
    dropout=0.1,
    gradients_norm=5,
)

def calculate_time(function):
  def inner(*args,**kwargs):
    begin=time.time()
    function(*args,**kwargs)
    end=time.time()
    print("This took {}s".format(end-begin))
  return inner

class Arthur:
  def __init__(self,flags):
    self.flags=flags
    self.Data=Data(flags)
    self.Brain=Brain(flags,self.Data.n_vocab)
    self.Pen=Pen(flags,self.Data.vocab_to_int,self.Data.int_to_vocab)

temp=0.5
probs=[80-60*temp,60-20*temp,50,40+20*temp,20+60*temp]
probs=[prob/250 for prob in probs]
np.random.choice(np.arange(1,6),1,p=probs)

class Pen:
  def __init__(self,flags,vocab_to_int,int_to_vocab):
    self.flags=flags
    self.vocab_to_int,self.int_to_vocab=vocab_to_int,int_to_vocab
    
  def write(self,Brain,words="Sherlock rubbed his",temperature=0.5,length=70):
    probs=[80-60*temp,60-20*temp,50,40+20*temp,20+60*temp]
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
    
    return ' '.join(words)

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
  def __init__(self,flags,n_vocab):
    super(Brain,self).__init__()
    self.flags=flags
    self.embedding= nn.Embedding(n_vocab, flags.embedding_size)
    self.lstm=nn.LSTM(flags.embedding_size, flags.lstm_size, batch_first=True, num_layers=flags.num_layers, dropout=flags.dropout)
    self.dense = nn.Linear(flags.lstm_size, n_vocab)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.flags.lr)
    self.loss_value=0
    self.to(flags.device)
    
  # Forward Pass
  def forward(self, x, prev_state):
      embed = self.embedding(x)
      output, state = self.lstm(embed, prev_state)
      logits = self.dense(output)
      return logits, state

  # ZeroState Init
  def zero_state(self,batch_size):
        return (torch.zeros(self.flags.num_layers*(2 if self.flags.is_bidirectional else 1), batch_size, self.flags.lstm_size),
                torch.zeros(self.flags.num_layers*(2 if self.flags.is_bidirectional else 1), batch_size, self.flags.lstm_size))
  
  #Train Function
  @calculate_time
  def Train(self,epochs,Data):
    self.train()
    for epoch in tqdm(range(epochs)):
      start=time.time()
      batches=Data.get_batches()
      state_h, state_c = self.zero_state(self.flags.batch_size)
      
      
      # Transfer data to GPU
      state_h = state_h.to(self.flags.device)
      state_c = state_c.to(self.flags.device)
      
      for x, y in (batches):
        
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
      print('Epoch: {}/{}'.format(epoch, epochs),'Loss: {}'.format(self.loss_value),'Time Taken : {}'.format(time.time()-start))