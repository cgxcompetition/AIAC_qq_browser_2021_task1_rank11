export app="$(
    cd -- "$(dirname "$0")" > /dev/null 2>&1
    pwd -P
)"
export data_path=$app/data

cd $app

python $app/ensemble.py