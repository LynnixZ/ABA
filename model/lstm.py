import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd,  n_layer, hidden_size=600,dropout=0.0):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden_size, n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)                             

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, emb_size]
        output, hidden = self.lstm(emb, hidden)  # output: [batch, seq_len, hidden_size]
        logits = self.fc(output)  # [batch, seq_len, vocab_size]
        return logits, hidden


    def step(self, token, hidden):
        
        emb = self.embedding(token)  # [batch, 1, n_embd]
        output, hidden = self.lstm(emb, hidden)  # output: [batch, 1, hidden_size]
        logits = self.fc(output)  # [batch, 1, vocab_size]
        return logits, hidden

    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        
        self.eval()
                
        with torch.no_grad():
                                   
            logits, hidden = self.forward(x, hidden=None)
                                   
            generated = x.clone()  # shape: [batch, seq_len]

                                         
            for _ in range(max_new_tokens):
                                        
                last_token = generated[:, -1:].clone()  # [batch, 1]
                                    
                logits, hidden = self.step(last_token, hidden)  # logits: [batch, 1, vocab_size]
                                   
                logits = logits[:, -1, :]  # [batch, vocab_size]
                                             
                scaled_logits = logits / temperature

                                            
                if top_k is not None:
                                        
                    values, indices = torch.topk(scaled_logits, top_k, dim=-1)
                                                          
                    filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                    filtered_logits.scatter_(1, indices, scaled_logits.gather(1, indices))
                    scaled_logits = filtered_logits

                        
                probs = torch.softmax(scaled_logits, dim=-1)  # [batch, vocab_size]
                                             
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                                       
                generated = torch.cat((generated, next_token), dim=1)  # [batch, seq_len+step]

            return generated