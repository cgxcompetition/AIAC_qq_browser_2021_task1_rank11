#### 描述
- 'data' 组织方提供的data目录，主要包括原始的预训练BERT模型和pointwise, pairwise, test等数据集
- 'save' 
    - 'data' 训练所需处理后的输入数据集
    - 'pretrain' 预训练模型存储路径
- 'src' 比赛源码
    - 'data_process' 数据集预处理脚本归档
    - '*.py' 基于提供baseline代码的修改代码

#### 代码执行逻辑
1. 环境部署: 参考src中组委会提供的教程，pip install -r src/requirements.txt
2. 数据copy: 将比赛数据集和预训练模型等材料存放在data目录中
3. 在save目录下新建两个文件夹data和pretrain
4. 进入到src/data_process目录，依次执行：
```shell script
    CUDA_VISIBLE_DEVICES=0 python pairwise_data_process.py
    CUDA_VISIBLE_DEVICES=0 python point_data_process.py
    CUDA_VISIBLE_DEVICES=0 python test_data_process.py
    CUDA_VISIBLE_DEVICES=0 python tag_list_data_process.py
```
5. 进入到src目录下，开始执行预训练：
```shell script
    CUDA_VISIBLE_DEVICES=0 python pre_train.py
```
6. 预训练完毕之后再save目录下新建ft_fold1~ft_fold5五个目录，同时把pretrain目录下的最新的checkpoint文件和最新的ckpt文件copy到这五个目录中。
7. 修改config_fold1~config_fold5配置文件，将pretrain-savedmodel-path路径修改和第6步一致，接着继续执行ft训练：
```shell script
    CUDA_VISIBLE_DEVICES=0 python train_fold1.py
    CUDA_VISIBLE_DEVICES=0 python train_fold2.py
    CUDA_VISIBLE_DEVICES=0 python train_fold3.py
    CUDA_VISIBLE_DEVICES=0 python train_fold4.py
    CUDA_VISIBLE_DEVICES=0 python train_fold5.py
```
8. 在训练完毕之后，修改对应的config.py文件中的ckpt-file参数，执行推理
```shell script
    CUDA_VISIBLE_DEVICES=0 python inference_fold1.py
    CUDA_VISIBLE_DEVICES=0 python inference_fold2.py
    CUDA_VISIBLE_DEVICES=0 python inference_fold3.py
    CUDA_VISIBLE_DEVICES=0 python inference_fold4.py
    CUDA_VISIBLE_DEVICES=0 python inference_fold5.py
```
9. 结果融合
```shell script
    python ensemble.py
```
10. 将result.json压缩成result.zip文件之后即可提交
11. 采用同样的发送采用baseline提供的tag_list文件，仍然可以训练一个新的五折模型，推理出来新的五折结果。
12. 将不同五折结果正则化之后，进行结果融合，作为最终的结果提交