find -name "*.*" | xargs dos2unix

export app="$(
    cd -- "$(dirname "$0")" > /dev/null 2>&1
    pwd -P
)"
export data_path=$app/data

cd $app

CUDA_VISIBLE_DEVICES=0 nohup bash $app/code_review_gpj/run_gpj.sh > gpj.log 2>&1 &

nohup bash $app/code_review_csj/run_csj.sh > csj.log 2>&1 &

nohup bash $app/code_review_xc/src/train.sh $data_path 1 > xc.log 2>&1 &