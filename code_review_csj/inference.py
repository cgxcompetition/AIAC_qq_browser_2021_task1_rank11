import json
from zipfile import ZIP_DEFLATED, ZipFile
import os
import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import CGXMultiModal



def main():
    parser.add_argument('--fold', default=0, type=int, help='kfold')
    parser.add_argument('--use_loss', default='mse', type=str, help='kfold')
    args = parser.parse_args()
    args.savedmodel_path = args.savedmodel_path.format(fold=args.fold, loss=args.use_loss)
    feature_parser = FeatureParser(args)
    files = args.test_b_file
    dataset = feature_parser.create_val_dataset(files, training=False, batch_size=args.test_batch_size)
    model = CGXMultiModal(args)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")

    vid_embedding = {}
    for inputs in dataset:
        convert_inputs = {
            'query_vid':inputs['vid'],
            'input_query_ids': inputs['input_ids'],
            'query_mask':inputs['mask'],
            'query_frames': inputs['frames'],
            'query_num_frames': inputs['num_frames'],
            'candidate_vid': inputs['vid'],
            'input_candidate_ids': inputs['input_ids'],
            'candidate_mask': inputs['mask'],
            'candidate_frames': inputs['frames'],
            'candidate_num_frames': inputs['num_frames']
        }
        _, embeddings, _, _, _ = model(convert_inputs, training=False)
        vids = inputs['vid'].numpy().astype(str)
        embeddings = embeddings.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            vid_embedding[vid] = embedding.tolist()

    result_json = os.path.join(args.savedmodel_path, args.output_json)
    result_zip = os.path.join(args.savedmodel_path, args.output_zip)
    print(result_json)
    with open(result_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(result_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(result_json)


if __name__ == '__main__':
    main()
