import gzip
import json

INPUT = "cp03/jawiki-country.json.gz"

OUTPUT = "cp03/out/20.txt"

with gzip.open(INPUT, 'rt', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス': 
            with open(OUTPUT, 'w', encoding='utf-8') as out_file: 
                out_file.write(data['text'])
            break