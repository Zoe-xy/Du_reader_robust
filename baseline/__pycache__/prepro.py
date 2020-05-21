# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import tokenization
import tensorflow as tf



def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
                c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


#
class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)


#
def read_squad_examples(input_file,vocab_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    #

    min_context_len=99999
    max_context_len=0
    sum_context_len = 0
    num_context=0

    min_answer_len=99999
    max_answer_len=0
    sum_answer_len=0
    num_answers=0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            # 分词,计算文档长度
            raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=False)
            sum_context_len+=len(raw_doc_tokens)
            min_context_len=min(min_context_len,len(raw_doc_tokens))
            max_context_len=max(max_context_len,len(raw_doc_tokens))
            num_context+=1

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:

                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k - 1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            assert k == len(raw_doc_tokens)

            # ChineseFullTokenizer
            tokenizer = ChineseFullTokenizer(vocab_file=vocab_file, do_lower_case=False)
            doc_tokens_C=tokenizer.tokenize(paragraph_text)

            print("BasicTokenizer length:%d"%(len(doc_tokens)))
            print("ChineseFullTokenizer length: %d"%(len(doc_tokens_C)))
            print(doc_tokens==doc_tokens_C)
            print(doc_tokens)
            print(doc_tokens_C)

            # # 计算answer长度
            # for qa in paragraph["qas"]:
            #     question_text = qa["question"]
            #     start_position = None
            #     end_position = None
            #     orig_answer_text = None
            #     # 开发集中某些问题答案不唯一
            #     for answer in qa["answers"]:
            #
            #         orig_answer_text = answer["text"]
            #
            #         if orig_answer_text not in paragraph_text:
            #             tf.logging.warning("Could not find answer")
            #         else:
            #             answer_offset = paragraph_text.index(orig_answer_text)
            #             answer_length = len(orig_answer_text)
            #             start_position = char_to_word_offset[answer_offset]
            #             end_position = char_to_word_offset[answer_offset + answer_length - 1]
            #             answer_len=(end_position-start_position+1)
            #
            #             num_answers+=1
            #             sum_answer_len+=answer_len
            #             min_answer_len=min(min_answer_len,answer_len)
            #             max_answer_len=max(max_answer_len,answer_len)
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        # actual_text = "".join(
                        #     doc_tokens[start_position:(end_position + 1)])

            # print('共%d个文档；平均长度为%.3f；最大长度为%d；最小长度%d；' %(
            #         num_context,sum_context_len/num_context,max_context_len,min_context_len))
            # print('共%d个答案；平均长度为%.3f；最大长度为%d；最小长度%d；'%(
            #     num_answers, sum_answer_len / num_answers, max_answer_len, min_answer_len))
    tf.logging.info("**********preprocess dataset complete!**********")

if __name__ == "__main__":
    train_file="../data/test.json"
    vocab_file="../bert/multi_cased_L-12_H-768_A-12/vocab.txt"
    read_squad_examples(train_file,vocab_file)




