

path = "/data/wufan/data/NERData/business_license/paddlenlp_data/20220822-ner-business-license-v2-train.jsonl"

import jsonlines
import random
result = []
with jsonlines.open(path) as reader:
    for line in reader:
        result.append(line)

random.seed(0)
random.shuffle(result)
n = len(result)
split = int(n*0.8)

with jsonlines.open("/data/wufan/data/NERData/business_license/paddlenlp_data/train.jsonl",'w') as w:
    for line in result[:split]:
        w.write(line)

with jsonlines.open("/data/wufan/data/NERData/business_license/paddlenlp_data/valid.jsonl",'w') as w:
    for line in result[split:]:
        w.write(line)