import pandas as np
import pandas as pd
import os


path_songs_name="midi2vec\edgelist"
path_song_embedding="midi2vec"


# Load the list of filenames
filenames_df=pd.read_csv(os.path.join(path_songs_name,"names.csv"))
filenames_df["id"]=filenames_df["id"].apply(lambda x: x.replace('=',''))
# Load the embeddings file
# Assuming each line starts with the filename followed by embeddings
embeddings = {}
with open(os.path.join(path_song_embedding,"embeddings.bin"), "r") as file:
    for line in file:
        parts = line.strip().split()
        # First part is the filename, rest are embeddings
        name = parts[0]
        vector = np.array(parts[1:], dtype=float)
        embeddings[name] = vector

# Match filenames with embeddings and prepare data for CSV
data_for_csv = []
for filename in filenames_df["id"]:
    if filename in embeddings:
        # Convert the numpy array to a list and prepend the filename
        row = [filename] + embeddings[filename].tolist()
        data_for_csv.append(row)

# Create a DataFrame and write to CSV
df = pd.DataFrame(data_for_csv)
df['singer'] = df[0].apply(lambda x: x.split('-')[1].replace('_', ' ')[:-1])
df['song'] = df[0].apply(lambda x: x.split('-')[2].replace('_', ' ')[1:])
df.drop(columns=0,inplace=True)
df.to_csv("matched_embeddings.csv", index=False, header=True)