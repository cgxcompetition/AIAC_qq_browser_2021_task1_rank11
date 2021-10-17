import logging
import os
import datetime
from pprint import pprint

from collections import deque
import tensorflow as tf
import numpy as np
from config import parser
from data_helper_cv import create_datasets
from metrics import Recorder
from model import CGXMultiModal
from util import test_spearmanr
import scipy


np.random.seed(5)
tf.random.set_seed(1234)


patience = 10
delta = 0.001

loss_history = deque(maxlen=patience + 1)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = 'logs/'
train_log_dir = LOG_DIR + current_time + '/train'
test_log_dir = LOG_DIR + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)


def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = CGXMultiModal(args)
    tb_callback.set_model(model)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    print(checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    # checkpoint
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    if args.use_loss == "mae":
        loss_object = tf.losses.mean_absolute_error
    if args.use_loss == "mse":
        loss_object = tf.losses.mean_squared_error
    if args.use_loss == "huber":
        loss_object = tf.losses.Huber(delta=0.6)
    loss_object1 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()


    @tf.function
    def cal_loss(query_labels, candidate_labels, score, predictions, query_label_predictions, candidate_label_predictions):
        score = tf.nn.sigmoid(3 * score)
        return 0.2 * loss_object(score, predictions) + loss_object1(query_labels, query_label_predictions) + loss_object1(candidate_labels, candidate_label_predictions)





    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        score = inputs['score']
        query_labels = inputs['query_labels']
        candidate_labels = inputs['candidate_labels']
        with tf.GradientTape() as tape:
            predictions, _, _, query_label_predictions, candidate_label_predictions = model(inputs, training=True)
            loss = cal_loss(query_labels, candidate_labels, score, predictions, query_label_predictions, candidate_label_predictions) * score.shape[-1] # convert mean back to sum
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, score, predictions)

    @tf.function
    def val_step(inputs):
        labels = inputs['labels']
        predictions, embeddings = model(inputs, training=False)
        loss = loss_object(labels, predictions) * labels.shape[-1]  # convert mean back to sum
        val_recorder.record(loss, labels, predictions)
        return vids, embeddings

    @tf.function
    def val_step1(inputs):
        vids = inputs['query_vid']
        query_labels = inputs['query_labels']
        candidate_labels = inputs['candidate_labels']
        scores = inputs['score']
        convert_inputs = {
            'query_vid':inputs['query_vid'],
            'input_query_ids': inputs['input_query_ids'],
            'query_mask':inputs['query_mask'],
            'query_frames': inputs['query_frames'],
            'query_num_frames': inputs['query_num_frames'],
            'candidate_vid': inputs['candidate_vid'],
            'input_candidate_ids': inputs['input_candidate_ids'],
            'candidate_mask': inputs['candidate_mask'],
            'candidate_frames': inputs['candidate_frames'],
            'candidate_num_frames': inputs['candidate_num_frames']
        }
        predictions, query_embeddings, _, query_label_predictions, candidate_label_predictions = model(convert_inputs, training=False)
        loss = cal_loss(query_labels, candidate_labels, scores, predictions, query_label_predictions, candidate_label_predictions) * scores.shape[-1]
        val_recorder.record(loss, scores, predictions)
        return vids, query_embeddings, predictions, scores

    # 6. training
    is_stop = False
    best_f1, best_spearman = 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        if is_stop:
            break
        for train_batch in train_dataset:
            checkpoint.step.assign_add(1)
            step = int(checkpoint.step.numpy())
            if step > args.total_steps:
                break
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                vid_pred = []
                label = []
                for val_batch in val_dataset:
                    _, _, pred, score = val_step1(val_batch)
                    vid_pred.extend(pred.numpy().tolist())
                    label.extend(score.numpy().tolist())

                # 8. test spearman correlation
                spearmanr = scipy.stats.spearmanr(vid_pred, label).correlation
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearmanr > 0.6:
                    checkpoint_manager.save(checkpoint_number=step)
                with test_summary_writer.as_default():
                    tf.summary.scalar("train_spearmanr", spearmanr, step=step)


                if spearmanr > best_spearman:
                    best_spearman = spearmanr

                loss_history.append(spearmanr)
                logging.info(f'{loss_history}')
                if len(loss_history) > patience:
                    # early stop
                    if loss_history.popleft() > max(loss_history):
                        is_stop = True # 停止当前epoch
                        logging.info(
                            f'\nEarly stopping. No improvement of validation spearman in the last {patience} epochs.')
                        break


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser.add_argument('--fold', default=0, type=int, help='kfold')
    parser.add_argument('--use_loss', default='mse', type=str, help='kfold')
    args = parser.parse_args()
    file = "save/data_v2/train_v2_{fold}.tfrecords"
    args.savedmodel_path = args.savedmodel_path.format(fold=args.fold, loss=args.use_loss)
    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)
    lst = [0, 1, 2, 3, 4]
    lst.remove(args.fold)
    args.train_kfold_file = file.format(fold=str(lst))
    args.val_kfold_file = file.format(fold=args.fold)
    args.total_steps = 100000
    args.bert_total_steps = 100000
    args.bert_lr = 3e-5
    args.lr = 0.00008
    args.warmup_steps = 2000
    args.bert_warmup_steps = 2000
    args.batch_size = 55
    args.val_batch_size = 128
    args.eval_freq = 5000
    args.max_to_keep = 4
    pprint(vars(args))


    train(args)


if __name__ == '__main__':
    main()
