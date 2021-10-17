echo "Start run csj_pipeline"

cd $app/code_review_csj

echo "Start to process finetune data"
python pairwise_data_process.py  >> process.log

echo "Start to pretrain model"
CUDA_VISIBLE_DEVICES=2 python pre_train.py  > pretrain_cross_modal.log 2>&1

echo "Start to finetune cross_modal model mse and mae"
bash run_pipeline_mae.sh 20210906 mae > finetune_mae.log 2>&1  &
bash run_pipeline_mse.sh 20210906 mse > finetune_mse.log 2>&1  &
wait

echo "Start to calculate five fold result"
python cal_kfold_all.py --use_loss mse
python cal_kfold_all.py --use_loss mae

echo "Over"
