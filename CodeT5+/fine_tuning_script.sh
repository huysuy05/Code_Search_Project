lang=javascript
pathTrainTestData="../" #Raw jsonl data directory
cacheDir="cache/$lang" #Directory to load cached data

mkdir -p $cacheDir

python3 tune_codet5p_seq2seq.py \
        --save-dir saved_models/$lang/ \
        --load Salesforce/codet5p-770m \
        --cache-data $cacheDir \
        --raw-data $pathTrainTestData/datasets/$lang/train.jsonl \
        --epochs 2 \
        --grad-acc-steps 2 \
        --max-source-len 256 \
        --max-target-len 128 \
        --batch-size-per-replica 16 \
        --lr 2e-5 \

