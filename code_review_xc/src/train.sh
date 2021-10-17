export app="$(
    cd -- "$(dirname "$0")" > /dev/null 2>&1
    pwd -P
)"

export data_path=$app/data

cd $app

data_path=$1
GPU=$2

rm -rf ../data
ln -s ${data_path} ../data

pip install -r requirements.txt

cd data_process
CUDA_VISIBLE_DEVICES=${GPU} python pairwise_data_process.py
CUDA_VISIBLE_DEVICES=${GPU} python point_data_process.py
CUDA_VISIBLE_DEVICES=${GPU} python test_data_process.py
CUDA_VISIBLE_DEVICES=${GPU} python tag_list_data_process.py

cd ..
CUDA_VISIBLE_DEVICES=${GPU} python pre_train.py

mkdir -p ../save/ft_fold1/20211001
mkdir -p ../save/ft_fold2/20211001
mkdir -p ../save/ft_fold3/20211001
mkdir -p ../save/ft_fold4/20211001
mkdir -p ../save/ft_fold5/20211001

cp -rf ../save/pretrain/20211001/checkpoint ../save/ft_fold1/20211001
cp -rf ../save/pretrain/20211001/checkpoint ../save/ft_fold2/20211001
cp -rf ../save/pretrain/20211001/checkpoint ../save/ft_fold3/20211001
cp -rf ../save/pretrain/20211001/checkpoint ../save/ft_fold4/20211001
cp -rf ../save/pretrain/20211001/checkpoint ../save/ft_fold5/20211001

cp -rf ../save/pretrain/20211001/ckpt-60000* ../save/ft_fold1/20211001
cp -rf ../save/pretrain/20211001/ckpt-60000* ../save/ft_fold2/20211001
cp -rf ../save/pretrain/20211001/ckpt-60000* ../save/ft_fold3/20211001
cp -rf ../save/pretrain/20211001/ckpt-60000* ../save/ft_fold4/20211001
cp -rf ../save/pretrain/20211001/ckpt-60000* ../save/ft_fold5/20211001

CUDA_VISIBLE_DEVICES=${GPU} python train_fold1.py
CUDA_VISIBLE_DEVICES=${GPU} python train_fold2.py
CUDA_VISIBLE_DEVICES=${GPU} python train_fold3.py
CUDA_VISIBLE_DEVICES=${GPU} python train_fold4.py
CUDA_VISIBLE_DEVICES=${GPU} python train_fold5.py

CUDA_VISIBLE_DEVICES=${GPU}  python inference_fold1.py
CUDA_VISIBLE_DEVICES=${GPU}  python inference_fold2.py
CUDA_VISIBLE_DEVICES=${GPU}  python inference_fold3.py
CUDA_VISIBLE_DEVICES=${GPU}  python inference_fold4.py
CUDA_VISIBLE_DEVICES=${GPU}  python inference_fold5.py

python ensemble.py
cp xc_result.json ../../output