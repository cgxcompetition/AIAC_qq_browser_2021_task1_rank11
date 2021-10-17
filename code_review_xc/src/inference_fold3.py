import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from config_fold3 import parser
from data_helper import FeatureParser
from model import CGXMultiModal


def main():
    args = parser.parse_args()
    feature_parser = FeatureParser(args)
    files = args.test_b_file
    dataset = feature_parser.create_val_dataset(files, training=False, batch_size=args.test_batch_size)
    model = CGXMultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")

    vid_embedding = {}
    for inputs in dataset:
        convert_inputs = {
            "query_vid": inputs['vid'],
            "input_query_ids": inputs['input_ids'],
            "query_mask": inputs['mask'],
            "query_asr_input_ids": inputs['asr_input_ids'],
            "query_asr_mask": inputs['asr_mask'],
            "query_frames": inputs['frames'],
            "query_num_frames": inputs['num_frames'],
            "candidate_vid": inputs['vid'],
            "input_candidate_ids": inputs['input_ids'],
            "candidate_mask": inputs['mask'],
            "candidate_asr_input_ids": inputs['asr_input_ids'],
            "candidate_asr_mask": inputs['asr_mask'],
            "candidate_frames": inputs['frames'],
            "candidate_num_frames": inputs['num_frames']
        }
        _, embeddings, _, _, _ = model(convert_inputs, training=False)
        vids = inputs['vid'].numpy().astype(str)
        embeddings = embeddings.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            vid_embedding[vid] = embedding.tolist()
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)


if __name__ == '__main__':
    main()
