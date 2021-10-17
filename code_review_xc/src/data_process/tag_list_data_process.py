import os
from pathlib import Path

import tensorflow as tf
from tqdm.notebook import tqdm

pointwise_data_path = "../../data/pointwise"
pairwise_data_path = "../../data/pairwise"
processed_data_path = "../../save/data/tag_list_12000.txt"

feature_map = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "title": tf.io.FixedLenFeature([], tf.string),
    "frame_feature": tf.io.VarLenFeature(tf.string),
    "tag_id": tf.io.VarLenFeature(tf.int64),
    "category_id": tf.io.VarLenFeature(tf.int64),
    "asr_text": tf.io.FixedLenFeature([], tf.string),
}

tag_list, id_list = [], []
dt_list = []
for p in tqdm(Path(pointwise_data_path).glob("*.tfrecords")):
    dataset = tf.data.TFRecordDataset(str(p))
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
    for a in tqdm(dataset):
        a["tag_id"] = tf.sparse.to_dense(a["tag_id"]).numpy()
        a["id"] = a["id"].numpy().decode()
        tag_list.append(a["tag_id"])
        id_list.append(a["id"])
        dt_list.append(a)

tag_count = {}
for item in tqdm(dt_list):
    tag_list = item["tag_id"]
    for tag in tag_list:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1

tag_list, id_list = [], []
dt_list = []
for p in tqdm(Path(pairwise_data_path).glob("*.tfrecords")):
    dataset = tf.data.TFRecordDataset(str(p))
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
    for a in tqdm(dataset):
        a["tag_id"] = tf.sparse.to_dense(a["tag_id"]).numpy()
        a["id"] = a["id"].numpy().decode()
        tag_list.append(a["tag_id"])
        id_list.append(a["id"])
        dt_list.append(a)

for item in tqdm(dt_list):
    tag_list = item["tag_id"]
    for tag in tag_list:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1

print(len(tag_count.keys()))

result = sorted(tag_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

with open(processed_data_path, "w", encoding="utf-8") as f:
    for tag in result[:12000]:
        f.write(str(tag[0]) + "\n")
