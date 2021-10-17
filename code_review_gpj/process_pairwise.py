from pathlib import Path

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

record_file = "save/data/train_v2.tfrecords"
pairwise_records_path = "../data/pairwise"
labels_path = "../data/pairwise/label.tsv"

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
for p in tqdm(Path(pairwise_records_path).glob("*.tfrecords")):
    dataset = tf.data.TFRecordDataset(str(p))
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map))
    for a in tqdm(dataset):
        a["category_id"] = tf.sparse.to_dense(a["category_id"]).numpy()
        a["frame_feature"] = tf.io.decode_raw(
            tf.sparse.to_dense(a["frame_feature"]), out_type=tf.float16
        ).numpy()
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
    {
        "id": id_list,
        "category_id": cat_list,
        "title": title_list,
        "tag_id": tag_list,
        "asr_text": text_list,
        "frame_feature": frame_list
    }
).set_index('id')

base_df = pd.read_csv(labels_path,
                      sep="\t",
                      names=["query_vid", "candidate_vid", "score"]
                      )

query_feature_df = pd.DataFrame(
    {
        "query_category_id": cat_list,
        "query_title": title_list,
        "query_tag_id": tag_list,
        "query_asr_text": text_list,
        "query_vid": id_list,
        "query_frame_feature": frame_list
    }
)
query_feature_df["query_vid"] = query_feature_df["query_vid"].astype(int)
query_df = base_df
new_df = query_df.merge(query_feature_df, on=["query_vid"], how="left")

candidate_feature_df = pd.DataFrame(
    {
        "candidate_category_id": cat_list,
        "candidate_title": title_list,
        "candidate_tag_id": tag_list,
        "candidate_asr_text": text_list,
        "candidate_vid": id_list,
        "candidate_frame_feature": frame_list
    }
)
candidate_feature_df["candidate_vid"] = candidate_feature_df["candidate_vid"].astype(int)
new_df = new_df.merge(candidate_feature_df, on=["candidate_vid"], how="left")

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




def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))




def train_feature(record):
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

with tf.io.TFRecordWriter(record_file) as writer:
    for rowid, record in new_df.iterrows():
        tf_example = tf.train.Example(features=train_feature(record))
        writer.write(tf_example.SerializeToString())