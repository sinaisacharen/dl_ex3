import pandas as pd
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class LyricsDataset(Dataset):
    """
    Dataset class for lyrics data. This class takes input vectors and target indices 
    and provides them as PyTorch tensors.
    """
    def __init__(self, input_vectors, target_indices):
        self.input_vectors = input_vectors
        self.target_indices = target_indices

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        return torch.tensor(self.input_vectors[idx]), torch.tensor(self.target_indices[idx])

class LyricsGenerator(nn.Module):  
    """
    LSTM-based model for lyrics generation. This model uses an embedding layer followed by 
    an LSTM and a fully connected layer to predict the next word in the sequence.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LyricsGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor containing word indices.
            hidden (tuple): Hidden state and cell state for the LSTM.
            
        Returns:
            out (torch.Tensor): Output predictions from the fully connected layer.
            hidden (tuple): Updated hidden state and cell state.
        """
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state and cell state for the LSTM.
        
        Args:
            batch_size (int): Batch size for the hidden states.
            
        Returns:
            tuple: Initialized hidden state and cell state.
        """
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

def generate_text(seed_text, model, max_length, vocab_size, word_to_idx, idx_to_word,word2vec):
    """
    Generates text using the trained LSTM model starting from a seed text.
    
    Args:
        seed_text (str): Initial text to start the generation.
        model (LyricsGenerator): Trained lyrics generation model.
        max_length (int): Maximum length of the generated text.
        vocab_size (int): Size of the vocabulary.
        word_to_idx (dict): Dictionary mapping words to their indices.
        idx_to_word (dict): Dictionary mapping indices to their words.
        word2vec (gensim.models.Word2Vec): Pre-trained Word2Vec model.
        
    Returns:
        str: 
    """
    model.eval()
    words = seed_text.lower().split()
    input_indices = [word2vec.wv[word] for word in words]
    x = torch.tensor([input_indices])  # Add batch dimension

    state_h, state_c = model.init_hidden(1)  # Initialize hidden state with batch size 1

    generated_words = words.copy()
    for _ in range(max_length):
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0, -1]  # Get the last time step
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_idx = np.random.choice(len(last_word_logits), p=p)
        new_word = idx_to_word[word_idx]
        generated_words.append(new_word)

        # Update x for the next prediction
        input_text=generated_words[-5:]
        input_indices = [word2vec.wv[word] for word in input_text]
        x = torch.tensor([input_indices])  # Add batch dimension

    #edit the final lyrics - take the first eos or last &
    final_lyrics=[]
    if 'eos' in generated_words:
        # Find the index of 'eos' and take all words before it
        index = generated_words.index('eos')
        final_lyrics.append(generated_words[:index])
    if '&' in generated_words:
        # If there is no 'eos', take all the words
        indecies = generated_words.index('&')
        last_index = len(generated_words) - 1 - generated_words[::-1].index('&')
        final_lyrics.append(generated_words[:last_index])
    else:
        final_lyrics.append(generated_words)

    return ' '.join(generated_words)
