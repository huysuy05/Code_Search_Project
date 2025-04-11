lang=go
pathTrainTestData="../"
mkdir -p $pathTrainTestData/saved_models/$lang/
python run_mac.py \
    --output_dir $pathTrainTestData/saved_models/$lang/ \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file $pathTrainTestData/datasets/$lang/train.jsonl \
    --eval_data_file $pathTrainTestData/datasets/$lang/test.jsonl \
    --codebase_file $pathTrainTestData/datasets/$lang/codebase.jsonl \
    --num_train_epochs 3 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456