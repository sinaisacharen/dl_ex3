import pandas as pd
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import LyricsGenerator, LyricsDataset,generate_text
import os
from sklearn.model_selection import train_test_split
import mido
from tools import low_case_name_file
os.environ["OMP_NUM_THREADS"] = "1"  # Limit the number of threads to 1
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

#editor_parameter
embedding_dim = 10 #entities per word
hidden_dim = 100
num_layers = 2
batch_size=5 
sequence_length=10 #lenght of the input of the model
num_epochs = 10

#1. Load your dataset
data = pd.read_csv('lyrics_train_set.csv')
data=data.iloc[:20,:]
sentences = []

midi_folder = "midi_files"
melody_embeddings = []

#reanme the midi files to low case letters



for lyrics in data['lyrics']:
    # Lowercase and remove characters between parentheses
    cleaned_lyrics = lyrics.lower()
    cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)
    
    # Split into words, then add 'eos' at the end
    words = cleaned_lyrics.split()
    words.append('eos')  # Append 'eos' as the end-of-sentence marker for the entire lyrics
    sentences.append(words)


#2. Embedding

#word2vect
word2vec = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
word2vec.train(sentences, total_examples=len(sentences), epochs=10)

word_to_idx = {word: idx for idx, word in enumerate(word2vec.wv.index_to_key)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
weights = torch.FloatTensor(word2vec.wv.vectors)



# Create input-output pairs
input_sequences = []
target_sequences = []

for sentence in sentences:
    if len(sentence) >= sequence_length:
        for i in range(len(sentence) - sequence_length):
            input_sequences.append(sentence[i:i + sequence_length])
            target_sequences.append(sentence[i + 1:i + sequence_length + 1])

input_vectors = [[word2vec.wv[word] for word in sequence] for sequence in input_sequences]
target_indices = [[word_to_idx[word] for word in sequence] for sequence in target_sequences]

## 3.ml model
#split the dataset to train-test
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
input_vectors, target_indices, test_size=0.2, random_state=42)

# Create dataset and dataloader
train_dataset = LyricsDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_dataset = LyricsDataset(val_inputs, val_targets)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

#Parmas
vocab_size = len(word_to_idx)

model = LyricsGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
model.embedding.weight = nn.Parameter(weights)
model.embedding.weight.requires_grad = False  # Freeze the embeddings

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_batch, target_batch in train_loader:
        current_batch_size = input_batch.size(0)  # Get the current batch size
        hidden = model.init_hidden(current_batch_size)  # Initialize hidden state for the current batch size

        optimizer.zero_grad()
        output, hidden = model(input_batch, hidden)
        output = output.view(-1, vocab_size)
        loss = criterion(output, target_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        hidden = tuple([h.detach() for h in hidden])  # Detach hidden states properly

    print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}')

    # Validation phase (as previously defined)
 
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to track gradients for validation
        for input_batch, target_batch in val_loader:
            current_batch_size = input_batch.size(0)  # Get the current batch size
            hidden = model.init_hidden(current_batch_size)  # Initialize hidden state for the current batch size
            output, hidden = model(input_batch, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion(output, target_batch.view(-1))
            val_loss += loss.item()

    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}')

    # Generate text after each epoch
    seed_text = "hey"  # Change as desired
    generated_text = generate_text(seed_text, model, 50, vocab_size, word_to_idx, idx_to_word,word2vec)  # Adjust length as needed
    print(f"Generated Text after Epoch {epoch+1}: {generated_text}")
