#!/bin/bash
export PYTHONIOENCODING=utf-8

for i in {0..3}
	 do
	echo "num${i} init_checkpoint is used"
	CUDA_VISIBLE_DEVICES=6 python baseline/run_squad.py \
						  --bert_model roberta \
                          --vocab_file output/vocab.txt \
                          --config_file output/bert_config.json \
                          --init_checkpoint output_1_1/pytorch_model${i}.bin \
						  --mode "eval" \
                          --predict_batch_size 128 \
						  --do_predict  \
					      --predict_file data/test1.json \
                          --eval_file data/dev.json \
                          --max_answer_length 30 \
						  --max_seq_length 512 \
                          --doc_stride 128 \
                          --output_dir output_1_1
                          $@
done 
