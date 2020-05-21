#!/bin/bash
export PYTHONIOENCODING=utf-8

CUDA_VISIBLE_DEVICES=2 python baseline/run_squad.py \
						  --bert_model roberta \
                          --vocab_file output_large/vocab.txt \
                          --config_file output_large/bert_config.json \
                          --init_checkpoint output_large/pytorch_model.bin \
						  --mode "predict" \
                          --predict_batch_size 32 \
						  --do_predict  \
					      --predict_file data/test1.json \
                          --eval_file data/dev.json \
                          --max_answer_length 512 \
						  --max_seq_length 512 \
                          --doc_stride 128 \
                          --output_dir output
                          $@

