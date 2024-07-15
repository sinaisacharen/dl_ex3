import sys
import time
import logging
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import math
import itertools
from gensim.models import KeyedVectors

os.getcwd()
model = KeyedVectors.load('midi2vec/embeddings.bin', mmap='r')
vectors = {}
for word in model.index_to_key:
    vectors[word] = model[word]

df = pd.DataFrame.from_dict(vectors, orient='index')

# Save to CSV
df.to_csv('embeddings.csv')

# Add Midi2Vec framework to Python working directory
sys.path.append('../')

from data_loading import MidiDataLoader
from midi_to_dataframe import NoteMapper
from pipeline import GenerativePipeline
from optimization import BruteForce
from evaluation import F1Evaluator, LossEvaluator