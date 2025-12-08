import gensim
from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
 path,
 binary = True
 )


analogy = model.most_similar(
 positive=["Spain", "Athens"], 
 negative=["Madrid"],
 topn=10)

print(analogy)