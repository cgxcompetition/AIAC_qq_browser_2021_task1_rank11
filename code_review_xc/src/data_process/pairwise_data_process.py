import os
import re
from pathlib import Path

import jieba
import jieba.analyse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

data_path = "../../data/pairwise"
processed_train_data_path = "../../save/data/pairwise_train_fold_{index}.tfrecords"
processed_val_data_path = "../../save/data/pairwise_val_fold_{index}.tfrecords"
pairwise_label_path = "../../data/pairwise/label.tsv"

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
     "frame_feature": frame_list})
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

base_df = pd.read_csv(pairwise_label_path, sep="\t", names=["query_vid", "candidate_vid", "score"])
print(base_df)

query_feature_df = feature_df.rename(
    columns={"id": "query_vid", "category_id": "query_category_id", "title": "query_title", "tag_id": "query_tag_id",
             "new_text": "query_asr_text", "frame_feature": "query_frame_feature"})
query_feature_df["query_vid"] = query_feature_df["query_vid"].astype(int)
query_feature_df = query_feature_df.drop(labels="asr_text", axis=1)
print(query_feature_df.info())
print(query_feature_df)

candidate_feature_df = feature_df.rename(
    columns={"id": "candidate_vid", "category_id": "candidate_category_id", "title": "candidate_title",
             "tag_id": "candidate_tag_id", "new_text": "candidate_asr_text",
             "frame_feature": "candidate_frame_feature"})
candidate_feature_df["candidate_vid"] = candidate_feature_df["candidate_vid"].astype(int)
candidate_feature_df = candidate_feature_df.drop(labels="asr_text", axis=1)
print(candidate_feature_df.info())
print(candidate_feature_df)

query_df = base_df
print(query_df.info())

new_df = query_df.merge(query_feature_df, on=["query_vid"], how="left")
print(new_df)

new_df = new_df.merge(candidate_feature_df, on=["candidate_vid"], how="left")
print(new_df)

print(new_df.info())

kf = KFold(n_splits=5, shuffle=True, random_state=365)


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
        "query_vid": _int64_feature(record["query_vid"]),
        "candidate_vid": _int64_feature(record["candidate_vid"]),
        "score": _float_feature(record["score"]),
        "query_category_id": _int64_list_feature(record["query_category_id"]),
        "candidate_category_id": _int64_list_feature(record["candidate_category_id"]),
        "query_tag_id": _int64_list_feature(record["query_tag_id"]),
        "candidate_tag_id": _int64_list_feature(record["candidate_tag_id"]),
        "query_title": _bytes_feature(record["query_title"].encode("utf-8")),
        "candidate_title": _bytes_feature(record["candidate_title"].encode("utf-8")),
        "query_asr_text": _bytes_feature(record["query_asr_text"].encode("utf-8")),
        "candidate_asr_text": _bytes_feature(record["candidate_asr_text"].encode("utf-8")),
        "query_frame_feature": _bytes_list_feature(record["query_frame_feature"]),
        "candidate_frame_feature": _bytes_list_feature(record["candidate_frame_feature"])
    }
    return tf.train.Features(feature=feature)


count = 0
for train_index, test_index in kf.split(new_df):
    print(count)
    x_train, x_test = new_df.iloc[train_index], new_df.iloc[test_index]
    print(x_train.info())
    print(x_test.info())
    train_record_file = processed_train_data_path.format(index=count)
    with tf.io.TFRecordWriter(train_record_file) as writer:
        for rowid, record in x_train.iterrows():
            tf_example = tf.train.Example(features=_train_feature(rowid, record))
            writer.write(tf_example.SerializeToString())

    test_record_file = processed_val_data_path.format(index=count)
    with tf.io.TFRecordWriter(test_record_file) as writer:
        for rowid, record in x_test.iterrows():
            tf_example = tf.train.Example(features=_train_feature(rowid, record))
            writer.write(tf_example.SerializeToString())
    count = count + 1

feature_map = {
    "query_vid": tf.io.FixedLenFeature([], tf.int64),
    "candidate_vid": tf.io.FixedLenFeature([], tf.int64),
    "score": tf.io.FixedLenFeature([], tf.float32),
    "query_tag_id": tf.io.VarLenFeature(tf.int64),
    "candidate_tag_id": tf.io.VarLenFeature(tf.int64),
    "query_category_id": tf.io.VarLenFeature(tf.int64),
    "candidate_category_id": tf.io.VarLenFeature(tf.int64),
    "query_title": tf.io.FixedLenFeature([], tf.string),
    "candidate_title": tf.io.FixedLenFeature([], tf.string),
    "query_asr_text": tf.io.FixedLenFeature([], tf.string),
    "candidate_asr_text": tf.io.FixedLenFeature([], tf.string),
    "query_frame_feature": tf.io.VarLenFeature(tf.string),
    "candidate_frame_feature": tf.io.VarLenFeature(tf.string)
}

count = 0
test_record_file = processed_val_data_path.format(index=count)

dataset = tf.data.TFRecordDataset(str(test_record_file))
dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
for a in tqdm(dataset):
    print(a["query_vid"].numpy())
    print(a["candidate_vid"].numpy())
    print(a["score"].numpy())
    print(tf.sparse.to_dense(a["query_category_id"]).numpy())
    print(tf.sparse.to_dense(a["query_tag_id"]).numpy())
    print(tf.sparse.to_dense(a["candidate_category_id"]).numpy())
    print(tf.sparse.to_dense(a["candidate_tag_id"]).numpy())
    print(a["query_title"].numpy().decode())
    print(a["candidate_title"].numpy().decode())
    print(a["query_asr_text"].numpy().decode())
    print(a["candidate_asr_text"].numpy().decode())
    print(tf.io.decode_raw(tf.sparse.to_dense(a["query_frame_feature"]), out_type=tf.float16).numpy())
    print(tf.io.decode_raw(tf.sparse.to_dense(a["candidate_frame_feature"]), out_type=tf.float16).numpy())
    print()
    print()
    count = count + 1
    if count > 20:
        break
