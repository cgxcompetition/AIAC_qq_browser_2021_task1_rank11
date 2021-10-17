import os
import re
from pathlib import Path

import jieba
import jieba.analyse
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm

data_path = "../../data/test_b"
processed_data_path = "../../save/data/test_b.tfrecords"

feature_map = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "title": tf.io.FixedLenFeature([], tf.string),
    "frame_feature": tf.io.VarLenFeature(tf.string),
    "tag_id": tf.io.VarLenFeature(tf.int64),
    "category_id": tf.io.VarLenFeature(tf.int64),
    "asr_text": tf.io.FixedLenFeature([], tf.string),
}

cat_list, tag_list, text_list, title_list, id_list, frame_list = [], [], [], [], [], []
dt_list = []
for p in tqdm(Path(data_path).glob("*.tfrecords")):
    dataset = tf.data.TFRecordDataset(str(p))
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
    for a in tqdm(dataset):
        a["category_id"] = tf.sparse.to_dense(a["category_id"]).numpy()
        a["frame_feature"] = tf.io.decode_raw(tf.sparse.to_dense(a["frame_feature"]), out_type=tf.float16).numpy()
        a["tag_id"] = tf.sparse.to_dense(a["tag_id"]).numpy()
        a["asr_text"] = a["asr_text"].numpy().decode()
        a["title"] = a["title"].numpy().decode()
        a["id"] = a["id"].numpy().decode()
        cat_list.append(a["category_id"])
        tag_list.append(a["tag_id"])
        text_list.append(a["asr_text"])
        title_list.append(a["title"])
        id_list.append(a["id"])
        frame_list.append(a["frame_feature"])
        dt_list.append(a)

feature_df = pd.DataFrame(
    {"id": id_list, "category_id": cat_list, "title": title_list, "tag_id": tag_list, "asr_text": text_list,
     "frame_feature": frame_list}).set_index("id")
print(feature_df)

asr_text_corpus = []
count = 0
for index, row in feature_df.iterrows():
    asr_text_corpus.append(row["title"] + " " + row["asr_text"])
    count = count + 1
print(len(asr_text_corpus))
print(asr_text_corpus[0])


# 创建停用词list
def _stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, "r", encoding="utf-8").readlines()]
    return stopwords


# 对句子进行分词
def _seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = _stopwordslist("stopwords.txt")
    outstr = ""
    for word in sentence_seged:
        word = word.strip()
        if word is not None and word not in stopwords:
            if word != "\t" and word != "":
                result = re.findall("[\u4e00-\u9fa5]", word)
                if result:
                    outstr += "".join(result)
    return outstr


processed_text_list = []
for text in tqdm(asr_text_corpus):
    tags = _seg_sentence(text)
    processed_text_list.append(tags)

print(len(processed_text_list))
print(processed_text_list[:10])

feature_df["new_text"] = processed_text_list


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    res = []
    for i in range(0, value.shape[0]):
        res.append(value[i].tostring())
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=res))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _train_feature(rowid, record):
    feature = {
        "id": _bytes_feature(rowid.encode("utf-8")),
        "title": _bytes_feature(record["title"].encode("utf-8")),
        "frame_feature": _bytes_list_feature(record["frame_feature"]),
        "tag_id": _int64_list_feature(record["tag_id"]),
        "category_id": _int64_list_feature(record["category_id"]),
        "asr_text": _bytes_feature(record["new_text"].encode("utf-8"))
    }
    return tf.train.Features(feature=feature)


with tf.io.TFRecordWriter(processed_data_path) as writer:
    for rowid, record in feature_df.iterrows():
        tf_example = tf.train.Example(features=_train_feature(rowid, record))
        writer.write(tf_example.SerializeToString())

cat_list, tag_list, text_list, title_list, id_list, frame_list = [], [], [], [], [], []
dt_list = []
count = 0


dataset = tf.data.TFRecordDataset(str(processed_data_path))
dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
for a in tqdm(dataset):
    a["category_id"] = tf.sparse.to_dense(a["category_id"]).numpy()
    a["frame_feature"] = tf.io.decode_raw(tf.sparse.to_dense(a["frame_feature"]), out_type=tf.float16).numpy()
    a["tag_id"] = tf.sparse.to_dense(a["tag_id"]).numpy()
    a["asr_text"] = a["asr_text"].numpy().decode()
    a["title"] = a["title"].numpy().decode()
    a["id"] = a["id"].numpy().decode()
    cat_list.append(a["category_id"])
    tag_list.append(a["tag_id"])
    text_list.append(a["asr_text"])
    title_list.append(a["title"])
    id_list.append(a["id"])
    frame_list.append(a["frame_feature"])
    dt_list.append(a)
    count = count + 1
    if count > 20:
        break

print(dt_list)
