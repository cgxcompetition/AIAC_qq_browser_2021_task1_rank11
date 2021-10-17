set -x
echo "Time$1"

for i in `seq 0 4`; do
  mkdir -p save/finetune/$1/$2/$i
  cp -r save/pretrain/$1/ckpt-50000* save/finetune/$1/$2/$i
  cp -r save/pretrain/$1/check* save/finetune/$1/$2/$i
  CUDA_VISIBLE_DEVICES=2 python train.py --fold $i --use_loss $2
  CUDA_VISIBLE_DEVICES=2 python inference.py --fold $i --use_loss $2
done
