## 필수 설치

- cudatoolkit=11.3.1
- cudnn=8.2.1
- scikit-learn
- tensorflow=2.6.0
- transformer=4.11.3
- python=3.7.11


## 학습 과정

- TFRecord 데이터 생성

```shell
python create_pretraining_data.py \
  --input_file=./sample2.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```


- 사전 학습 시작 (init_checkpoint 는 처음 학습할 경우 사용 x)

```shell
python run_pretraining.py \
  --input_file=tf_examples.tfrecord \
  --output_dir=pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=bert_config.json \
  --init_checkpoint=pretraining_output/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=90000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```


- 결과 화면

```
***** Eval results *****
  global_step = 90000
  loss = 0.55757356
  masked_lm_accuracy = 0.83186007
  masked_lm_loss = 0.5474623
  next_sentence_accuracy = 0.9925
  next_sentence_loss = 0.013958289
```

