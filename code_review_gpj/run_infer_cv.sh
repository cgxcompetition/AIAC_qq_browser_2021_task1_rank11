set -x
mkdir -p ./save/output/
for i in $(seq 0 4); do
    model_path=save/finetune/finetune_gpj/$i
    echo $model_path
    python inference_cv.py --savedmodel-path $model_path \
        --output-json ./save/output/fold${i}_testb.json
    COUNTER=$((COUNTER + 1))
done
