# import datetime
import logging
import os
from pprint import pprint
# os.environ['CUDA_VISIBLE_DEVICES']="0"

import tensorflow as tf
import numpy as np
np.random.seed(5)
tf.random.set_seed(1234)
from config import parser
from data_helper import create_pretrain_datasets
from model import CGXMultiModal
from pretrain_metrics import Recorder
from model import CGXMultiModal
from util import test_spearmanr

# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_pretrain_datasets(args)
    # 2. build model
    model = CGXMultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.pretrain_savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    # loss_object1 = tf.losses.mean_squared_error
    # loss_object2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()
    
    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        convert_inputs = {
            "query_vid": inputs['vid'],
            "input_query_ids": inputs['input_ids'],
            "query_mask": inputs['mask'],        
            "query_frames": inputs['frames'],
            "query_num_frames": inputs['num_frames'],
            "candidate_vid": inputs['vid'],
            "input_candidate_ids": inputs['input_ids'],
            "candidate_mask": inputs['mask'],            
            "candidate_frames": inputs['frames'],
            "candidate_num_frames": inputs['num_frames']
        }
        with tf.GradientTape() as tape:
            predictions, query_embeddings, candidate_embeddings, query_label_prediction, candidate_label_prediction = model(
                convert_inputs, training=True)
            # loss = cal_loss(query_labels, candidate_labels, score, predictions, query_label_prediction,
            #                 candidate_label_prediction) * query_labels.shape[-1]  # convert mean back to sum
            loss = loss_object(labels, query_label_prediction)*labels.shape[-1]
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, labels, query_label_prediction)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        score = tf.ones([labels.shape[0], 1], tf.float32)
        convert_inputs = {
            "query_vid": inputs['vid'],
            "input_query_ids": inputs['input_ids'],
            "query_mask": inputs['mask'],
            "query_frames": inputs['frames'],
            "query_num_frames": inputs['num_frames'],
            "candidate_vid": inputs['vid'],
            "input_candidate_ids": inputs['input_ids'],
            "candidate_mask": inputs['mask'],
            "candidate_frames": inputs['frames'],
            "candidate_num_frames": inputs['num_frames']
        }
        predictions, query_embeddings, candidate_embeddings, query_label_predictions, candidate_label_predictions = model(
            convert_inputs, training=False)
        loss = loss_object(labels, query_label_predictions) * labels.shape[-1]  # convert mean back to sum

        val_recorder.record(loss, labels, query_label_predictions)
        return vids, query_embeddings

    # 6. training
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch in train_dataset:
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                vid_embedding = {}
                for val_batch in val_dataset:
                    vids, embeddings = val_step(val_batch)
                    for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
                        vid = vid.decode('utf-8')
                        vid_embedding[vid] = embedding
                # 8. test spearman correlation
                spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearmanr > 0.55:
                    checkpoint_manager.save(checkpoint_number=step)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()
    # args.total_steps = 40_000
    # args.warmup_steps = 6_000
    # args.bert_total_steps = 80_000
    # args.bert_warmup_steps = 2_000
    if not os.path.exists(args.pretrain_savedmodel_path):
        os.makedirs(args.pretrain_savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
