export bert_model=./bert-base-german-cased
export bert_model_output=tmp/no-scare-balanced-base-german-cased/
export dataset=../training-data/no-scare-balanced

echo train $bert_model_output 
python run_classifier.py \
  --bert_model=$bert_model \
  --task_name=sentiment \
  --do_train \
  --data_dir=$dataset \
  --max_seq_length=256 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$bert_model_output 
exit
echo test $bert_model_output 
python test.py \
  --bert_model=$bert_model \
  --output_dir=$bert_model_output \
  --task_name=sentiment \
  --data_dir=$dataset \
  --max_seq_length=256 \
  --eval_batch_size=100 \
no-scare-balancedno-scare-balancedexit