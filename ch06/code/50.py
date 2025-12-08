import gensim
from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
 path,
 binary = True
 )

text = "United_States"

vec = model[text]
print(vec)