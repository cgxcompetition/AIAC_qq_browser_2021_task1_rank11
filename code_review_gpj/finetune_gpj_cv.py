import datetime
import logging
import os
from pprint import pprint

# from scipy.sparse.construct import rand

from config import parser
parser.add_argument('--use-loss', default='mse',type=str,help='loss')
parser.add_argument('--gpu-num',default='0',type=str,help='gpu')
args=parser.parse_args()
# args.total_steps = 100_000
# args.warmup_steps = 6_000
# args.bert_total_steps=100_000
# args.bert_warmup_steps=2_000
# os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_num

import numpy as np
np.random.seed(5)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(1234)
from sklearn.model_selection import KFold

from data_helper_cv import create_datasets
from metrics import Recorder
from model import CGXMultiModal
from util import test_spearmanr_cv

# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def train(args,train_idx,val_idx,val_vid_set):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args,train_idx,val_vid_set)
    # 2. build model
    model = CGXMultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    if args.use_loss == 'mae':
        loss_object = tf.losses.mean_absolute_error
    if args.use_loss == 'mse':
        loss_object = tf.losses.mean_squared_error        
    loss_object2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()

    @tf.function
    def cal_loss(query_labels, candidate_labels, score, predictions, query_label_predictions,
                 candidate_label_predictions):
        score = tf.nn.sigmoid(3 * score)
        return loss_object(score, predictions) + loss_object2(query_labels,
                                                                      query_label_predictions) + loss_object2(
            candidate_labels, candidate_label_predictions)

    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        inputs = inputs[1] # enumerate
        score = inputs['score']
        query_labels = inputs['query_labels']
        candidate_labels = inputs['candidate_labels']
        with tf.GradientTape() as tape:
            predictions, _, _, query_label_predictions, candidate_label_predictions = model(inputs, training=True)
            loss = cal_loss(query_labels, candidate_labels, score, predictions, query_label_predictions,
                            candidate_label_predictions) * score.shape[-1]  # convert mean back to sum
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, score, predictions)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        labels = inputs['labels']
        scores = tf.ones([labels.shape[0],1],tf.float32)
        convert_inputs = {
            'query_vid':inputs['vid'],
            'input_query_ids':inputs['input_ids'],
            'query_mask':inputs['mask'],
            'query_frames':inputs['frames'],
            'query_num_frames':inputs['num_frames'],
            'candidate_vid':inputs['vid'],
            'input_candidate_ids':inputs['input_ids'],
            'candidate_mask':inputs['mask'],
            'candidate_frames':inputs['frames'],
            'candidate_num_frames':inputs['num_frames']
        }
        predictions, query_embeddings, candidate_embeddings, _,_ = model(
            convert_inputs, training=False)
        loss = loss_object(scores,predictions)*labels.shape[-1]
        val_recorder.record(loss, scores, predictions)
        return vids,query_embeddings

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
                vid_embeddings = {}
                
                for val_batch in val_dataset:
                    vids,embeddings = val_step(val_batch)
                    for vid,embedding in zip(vids.numpy(),embeddings.numpy()):
                        vid=vid.decode('utf-8')
                        vid_embeddings['vid']=embedding                    
                # 8. test spearman correlation
                spearmanr = test_spearmanr_cv(vid_embeddings,args.annotation_file,val_idx)
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearmanr > 0.45:
                    checkpoint_manager.save(checkpoint_number=step)


def main(args):
    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)
    labels = pd.read_csv(args.annotation_file,sep='\t',header=None).astype(str)
    for i,(train_idx,val_idx) in enumerate(KFold(n_splits=5,shuffle=True,random_state=1).split(labels)):
        if i!=args.fold_num:
            continue
        val_labels = labels.iloc[val_idx, :]
        val_vid_set = set(val_labels[0]) | set(val_labels[1])
        train(args, train_idx, val_idx, val_vid_set)
    pprint(vars(args))

if __name__ == '__main__':
    main(args)
