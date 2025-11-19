N = 10

with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

size = len(lines) // N
for i in range(N):
        start = i * size
        end = (i + 1) * size if i < N - 1 else len(lines)
        with open(f"cp02/out/05/05_{i+1}.txt", "w", encoding="utf-8") as out:
            out.writelines(lines[start:end])
            
            
# split -l $(( $(wc -l < cp02/text/popular-names.txt)/10 )) 
# cp02/text/popular-names.txt cp02/out/05/split_
