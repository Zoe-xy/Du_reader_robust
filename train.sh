#!/bin/bash
export PYTHONIOENCODING=utf-8
# 设置cuda
# -z 判断 变量的值，是否为空
#if [ -z "$CUDA_VISIBLE_DEVICES" ];then
#    export CUDA_VISIBLE_DEVICES=0


# bool值传入空值才会是False，其余全为True
CUDA_VISIBLE_DEVICES=0 python  baseline/run_squad.py \
                          --bert_model bert \
						  --vocab_file bert/vocab.txt \
                          --config_file bert/bert_config.json \
                          --init_checkpoint bert/pytorch_model.bin \
                          --do_train  \
                          --train_file data/train.json \
                          --eval_file data/dev.json \
						  --da_file  data/processed_synponyms.json\
						  --train_batch_size 16 \
                          --gradient_accumulation_steps 16 \
						  --num_train_epochs 2 \
						  --max_checkpoint 4 \
						  --save_checkpoint_steps 2000 \
                          --max_seq_length 512 \
                          --doc_stride 128 \
						  --max_answer_length 30 \
                          --learning_rate 3e-5 \
                          --output_dir output_1_2 \
                          $@ 
