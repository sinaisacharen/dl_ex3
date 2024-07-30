"""
text similarity

https://ieeexplore.ieee.org/document/10493834

"""


from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define two sentences to compare
sentence1 = "The cat sits on the mat."
sentence2 = "A cat is sitting on a mat."

# Compute embeddings for both sentences
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# Compute cosine similarity between embeddings
similarity = util.pytorch_cos_sim(embedding1, embedding2)

print(f'Similarity Score: {similarity.item():.2f}')




from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are pets."
]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the TF-IDF matrix to a DataFrame for better readability
import pandas as pd

# Get feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame with the TF-IDF scores
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print(df_tfidf)
