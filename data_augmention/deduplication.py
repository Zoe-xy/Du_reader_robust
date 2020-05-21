#! -*- coding: utf-8 -*-
# 对生成的同义句去重

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout
from keras.layers import *
import json
import logging
import collections
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# 创建一个文件的handler
f_handler = logging.FileHandler("deduplication.log");
f_handler.setLevel(logging.INFO);
# 绑定格式
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s");
f_handler.setFormatter(fmt)
# 绑定一个handler
logger.addHandler(f_handler);




def main():
    filepath = "../data/generate.json"
    output_file = "../data/processed_synponyms.json"
    logger.info("start read_file")
    with open(filepath, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    logger.info("start deduplication")

    all_synponyms = []
    i = 0
    min_syn=99999
    for dict in input_data:
        qas_id = dict["qas_id"]
        question_text = dict["question_text"]
        synonyms_pre=dict["synonyms"]
        synonym_text=[]
        length=max(len(question_text)-4,4)
        tmp = question_text[:2].replace(r"+", "\\+")
        logger.info("tmp:" + tmp)
        # 如果英文或数字开头 第一个中文字符前的所有字符串为匹配段
        if tmp.encode('utf-8').isalnum():
            pos=[m.start() for m in  re.finditer(r'[\u4e00-\u9fa5]',question_text )]
            if len(pos)>0:
                tmp = question_text[:pos[0]]
                logger.info("tmp split " + tmp)

        for s in synonyms_pre:
            # s=s.replace(question_text,"")
            # if len(s)<length or s in synonym_text:
            #     continue
            # else:
            #
            #     synonym_text.append(s)

            # s=s.replace(question_text,"")
            # 转为utf-8再判断是否是英文  否则中文也会判断为True

            idx=[m.start() for m in re.finditer(tmp,s)]
            # 同义句中有重复句
            if len(idx)>1 and (idx[1]-idx[0])>length:
                s=s[:idx[1]]
                if s in synonym_text:
                    continue
            s=s.replace(r"\\","")
            synonym_text.append(s)
        synonyms = collections.OrderedDict()
        synonyms["qas_id"] = qas_id
        synonyms["question_text"] = question_text
        synonyms["synonyms"] = synonym_text
        if len(synonym_text)<=min_syn:
            min_syn=len(synonym_text)
            logger.info("min_syn: %d" %(min_syn))
        logger.info("num: %d" % (i))
        all_synponyms.append(synonyms)
        i += 1
    logger.info("min number of synonyms: %d" %(min_syn))
    logger.info("write to output_file")
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_synponyms, ensure_ascii=False, indent=4) + "\n")
    logger.info("writing to file complete")


if __name__ == "__main__":
    main()


