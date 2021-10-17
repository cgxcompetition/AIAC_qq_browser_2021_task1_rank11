echo $app
echo $data_path
cd $app/code_review_gpj

set -x
# 1. 处理数据集
# 数据集生成路径为"save/data/train_v2.tfrecords"
mkdir -p save/data/
python process_pairwise.py

# 2. 在pointwise数据上预训练
# 生成文件路径./save/pretrain/pretrain_gpj/
# 生成step 40000

python pre_train_gpj.py \
    --total-steps 40000 \
    --warmup-steps 6000 \
    --bert-total-steps 80000 \
    --bert-warmup-steps 2000 \
    --pretrain-savedmodel-path save/pretrain/pretrain_gpj \
    > pretrain.log 2>&1


# 3. 在pairwise数据集上五折finetune
# 生成./save/finetune/finetune_gpj/0 五折模型
bash run_finetune.sh
# 4. 五折推理
bash run_infer_cv.sh
# 5. 五折合并
python merge_folds_result.py 
