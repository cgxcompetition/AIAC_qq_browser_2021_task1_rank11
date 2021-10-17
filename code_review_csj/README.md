### [Multimodal Video Similarity Challenge](https://algo.browser.qq.com/)
#### [@CIKM 2021](https://www.cikm2021.org/analyticup) 
Refer to the official tensorflow version baseline

#### 1. data
store data generated after pairwise processing

#### 2. code description
- [config.py](config.py) contains all the configuration
- [data_helper.py](data_helper.py) handles data processing and parsing
- [data_helper_cv.py](data_helper_cv.py) handles data processing and parsing support kfold training
- [evaluate.py](evaluate.py) evaluates the model
- [inference.py](inference.py) generate the submitting file
- [pretrain_metrics.py](pretrain_metrics.py) outputs the metric when pretraining
- [metrics.py](metrics.py) outputs the metric when training
- [model.py](model.py) builds the cross-modal model
- [train.py](train.py) is the training entry
- [train_mae.py](train_mae.py) is the training entry
- [cal_kfold_all.py](cal_kfold_all.py) calculate kfold training result
- [run_pipeline_mae.sh](run_pipeline_mae.sh) train mae loss cross-modal model
- [run_pipeline_mse.sh](run_pipeline_mse.sh) train mse loss cross-modal model
- [run_csj.sh](run_csj.sh) end to end pipeline

#### 3. Run the complete pipeline
```bash
bash code_review_csj/run_csj.sh
```

#### 4. Process the pairwise data
```bash
python pairwise_data_process.py  >> process.log
```

#### 5. Pretrain
```bash
CUDA_VISIBLE_DEVICES=2 python pre_train.py  > pretrain_cross_modal.log
```

#### 6. Finetune Inference
```bash
bash run_pipeline_mae.sh 20210906 mae > finetune_mae.log 2>&1  &
bash run_pipeline_mse.sh 20210906 mse > finetune_mse.log 2>&1  &
```

#### 7. Calculate five fold result
```bash
python cal_kfold_all.py --use_loss mse
python cal_kfold_all.py --use_loss mae
```
