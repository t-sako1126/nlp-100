def n_grams(sequence, n):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
   
x = "paraparaparadise"
y = "paragraph"

x_bi = n_grams(x, 2)
y_bi = n_grams(y, 2)
X = set(x_bi)
Y = set(y_bi)

print("和集合:", X | Y)
print("積集合:", X & Y)
print("差集合:", X - Y)

print("'se' in X:", "se" in X)
print("'se' in Y:", "se" in Y)
