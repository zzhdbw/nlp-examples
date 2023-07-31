# -*- coding: utf-8 -*-            
# @Author : Zhihao Zhang
# @Time : 2023/7/29 11:42

from stanfordcorenlp import StanfordCoreNLP
import json
from tqdm import tqdm

stanfordNlp = StanfordCoreNLP("D:\stanford-corenlp-full-2018-02-27", lang='zh')


def process(mode, text_max_len):
    in_path = "./data/ChnSentiCorp/{}.tsv".format(mode)
    out_path = "./data/ChnSentiCorp/{}.json".format(mode)

    with open(in_path, 'r', encoding='utf-8') as r, open(out_path, 'w', encoding='utf-8') as w:
        for line in tqdm(r.readlines()[1:]):
            if "train" in in_path:
                label, text = line.strip().split("\t")
            else:
                _, label, text = line.strip().split("\t")
            words = list(stanfordNlp.word_tokenize(text[:text_max_len]))
            arcs = list(stanfordNlp.dependency_parse(text[:text_max_len]))

            sort_list = sorted(arcs, key=lambda x: x[2])
            dep_type, head, _ = zip(*sort_list)

            w.write(json.dumps({"text": text, "label": int(label), "words": words, "heads": head, "dep_type": dep_type},
                               ensure_ascii=False) + "\n")


process("train", 512)
process("dev", 512)
