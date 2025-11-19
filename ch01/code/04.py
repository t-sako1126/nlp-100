s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

s = s.replace(",", "").replace(".", "")
words = s.split()
target = [1, 5, 6, 7, 8, 9, 15, 16, 19]
result = {}

for i, word in enumerate(words):
    if (i + 1) in target:
        result[word[0]] = i + 1
    else:
        result[word[:2]] = i + 1
        
print(result)
