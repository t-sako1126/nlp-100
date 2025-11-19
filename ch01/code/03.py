s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

s = s.replace(",", "").replace(".", "")
words = s.split()

lengths = [len(word) for word in words]
print(lengths)