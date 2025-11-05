def n_grams(sequence, n):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

s = "I am an NLPer"

s_no_spaces = s.replace(" ", "")
print("文字tri_grams:", n_grams(s_no_spaces, 3))
print("単語bi_grams:", n_grams(s.split(), 2))
