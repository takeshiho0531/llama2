from .get_embeddings import main
from openai.embeddings_utils import cosine_similarity
import pandas as pd

df=pd.read_csv()

for column in ["text_1", "text_2"]:
    df[f"{column}_embedding"] = df[column].apply(main)

# create column of cosine similarity between embeddings
df["cosine_similarity"] = df.apply(
    lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
    axis=1,
)

