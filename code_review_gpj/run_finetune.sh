set -x
current='finetune_gpj'
for i in $(seq 0 4); do
    savedmodel_path=save/finetune/${current}/$i
    mkdir -p $savedmodel_path
    cp -r save/pretrain/pretrain_gpj/ckpt-* ${savedmodel_path}
    cp -r save/pretrain/pretrain_gpj/check* ${savedmodel_path}
    python finetune_gpj_cv.py \
        --fold-num $i \
        --train-record-pattern save/data/train_v2.tfrecords \
        --annotation-file $data_path/pairwise/label.tsv \
        --savedmodel-path ${savedmodel_path} \
        --total-steps 100000 \
        --warmup-steps 6000 \
        --bert-total-steps 100000 \
        --bert-warmup-steps 2000 \
        --use-loss mse         
done
