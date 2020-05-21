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


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    return s
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               tokens,
               start_position=None,
               end_position=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.tokens = tokens
    self.start_position = start_position
    self.end_position = end_position
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
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        # 增加了控制字符的判断
        if tokenization._is_control(c):
            continue
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
                c) :
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


#






def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    #
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            # 分词
            raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=False)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:
                # 增加控制字符的判断
                if tokenization._is_whitespace(c) or tokenization._is_control(c):
                    char_to_word_offset.append(k - 1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)

                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1
            print("k:%d;raw_doc_tokens:%d"%(k,len(raw_doc_tokens)))
            print("qas_id:%s"%(paragraph["qas"][0]["id"]))
            if (k!=len(raw_doc_tokens)):
                print("doc_tokens:")
                print(doc_tokens)

                print("raw_doc_tokens:")
                print(raw_doc_tokens)
            assert k == len(raw_doc_tokens)
            # 答案的basic token位置
            start_positions=[]
            end_positions=[]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                for answer in qa["answers"]:
                    orig_answer_text = answer["text"]

                    if orig_answer_text not in paragraph_text:
                        tf.logging.warning("Could not find answer")
                    else:
                        answer_offset = paragraph_text.index(orig_answer_text)
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue
                        start_positions.append(start_position)
                        end_positions.append(end_position)

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_positions,
                    end_position=end_positions)
                examples.append(example)
    tf.logging.info("**********read_squad_examples complete!**********")

    return examples

def convert_examples_to_features(examples, tokenizer, vocab_file,is_training=True):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  tokenizer = ChineseFullTokenizer(vocab_file=vocab_file, do_lower_case=False)
  input_features = []
  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)


    # 对doc进一步分词，
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    start_positions=[]
    end_positions=[]
    if is_training:
      for (orig_start_position,orig_end_start_position)in zip(example.start_position,example.end_position):
          tok_start_position = orig_to_tok_index[orig_start_position]
          if orig_end_start_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[orig_end_start_position + 1] - 1
          else:
            tok_end_position = len(all_doc_tokens) - 1
          (tok_start_position, tok_end_position) = _improve_answer_span(
              all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
              example.orig_answer_text)
          start_positions.append(tok_start_position)
          end_positions.append(tok_end_position)


    feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          tokens=all_doc_tokens,
          start_position=start_positions,
          end_position=end_positions)
    input_features.append(feature)
    unique_id+=1

  return input_features

if __name__ == "__main__":
    train_file = sys.argv[1]
    vocab_file = "../bert/multi_cased_L-12_H-768_A-12/vocab.txt"
    min_context_len=99999
    max_context_len=0
    sum_context_len = 0
    num_context=0
    min_answer_len=99999
    max_answer_len=0
    sum_answer_len=0
    num_answers=0

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)
    examples=read_squad_examples(input_file=train_file,is_training=True)
    features=convert_examples_to_features(examples,tokenizer,vocab_file=vocab_file,is_training=True)

    for feature in features:
        len_context=len(feature.tokens)
        sum_context_len+=len_context
        max_context_len=max(max_context_len,len_context)
        min_context_len=min(min_context_len,len_context)
        num_context+=1

        for (start,end) in zip(feature.start_position,feature.end_position):
            len_answer=end-start+1
            sum_answer_len+=len_answer
            max_answer_len=max(max_answer_len,len_answer)
            min_answer_len=min(min_answer_len,len_answer)
            num_answers+=1
    print("共 %d 个文档"%(num_context))
    print("平均长度:%.3f, 最长 %d , 最短 %d "%(sum_context_len/num_context,max_context_len,min_context_len))
    print("共 %d 个答案" % (num_answers))
    print("平均长度: %.3f , 最长 %d , 最短 %d " % (sum_answer_len / num_answers, max_answer_len, min_answer_len))






