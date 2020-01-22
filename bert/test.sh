export bert_model=./bert-base-german-cased
export bert_model_output=../models/paper-no-scare-balanced/bert
export dataset=../training-data/no-scare-balanced

python3 test.py \
  --bert_model=$bert_model \
  --output_dir=$bert_model_output \
  --task_name=sentiment \
  --data_dir=$dataset \
  --max_seq_length=256 \
  --eval_batch_size=100 \
  --fp16 

exit