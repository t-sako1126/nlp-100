import gensim
from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
 path,
 binary = True
 )

text1 = "United_States"
text2 = "U.S."

sim = model.similarity(text1, text2)
print(sim)