#! -*- coding: utf-8 -*-
# SimBERT base 基本例子

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
f_handler = logging.FileHandler("symbert.log");
f_handler.setLevel(logging.INFO);
# 绑定格式
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s");
f_handler.setFormatter(fmt)
# 绑定一个handler
logger.addHandler(f_handler);

maxlen = 32

# bert配置
config_path = './chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        sim=[]
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        # 正则化去掉标点 去重
        p=re.compile(r"[！？‘，；。?!',]")
        for ids in output_ids:
            str=tokenizer.decode(ids)
            str=p.sub(repl='',string=str)
            if str not in sim:
                sim.append(str)
        #     else:
        #         logger.info("generate :"+str)
        # # return [tokenizer.decode(ids) ]
        return sim

synonyms_generator = SynonymsGenerator(start_id=None,
                                       end_id=tokenizer._token_end_id,
                                       maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()

    return [r[i + 1] for i in argsort[:k]]


def read_squad(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    example = collections.OrderedDict()
    examples = []
    i = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                example[qas_id] = question_text
                # examples.append(example[qas_id])
                i += 1
    logger.info("共%d个example" % (i))

    return example


def main():
    filepath = "../data/train.json"
    output_file = "../data/generate.json"
    logger.info("start read_squad")
    examples = read_squad(filepath)

    logger.info("start generate synonyms")
    all_synponyms = []
    i = 0
    for key in examples.keys():
        qas_id = key
        question_text = examples[key]
        synonyms = collections.OrderedDict()
        synonym_text = gen_synonyms(question_text, n=100, k=8)
        synonyms["qas_id"] = qas_id
        synonyms["question_text"] = question_text
        synonyms["synonyms"] = synonym_text
        logger.info("num: %d" % (i))
        logger.info(synonyms)
        all_synponyms.append(synonyms)
        i += 1

    logger.info("write to output_file")
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_synponyms, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    main()


