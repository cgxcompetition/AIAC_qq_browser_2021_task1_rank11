import logging
import os
from pprint import pprint

import scipy
import tensorflow as tf

from config_fold5 import parser
from data_helper import create_datasets
from metrics import Recorder
from model import CGXMultiModal


def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
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
    loss_object1 = tf.losses.mean_squared_error
    loss_object2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()

    @tf.function
    def cal_loss(query_labels, candidate_labels, score, predictions, query_label_prediction,
                 candidate_label_prediction):
        change_score = tf.nn.sigmoid(3 * score)
        return loss_object1(change_score, predictions) + loss_object2(query_labels,
                                                                      query_label_prediction) + loss_object2(
            candidate_labels, candidate_label_prediction)

    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
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
        score = inputs['score']
        query_labels = inputs['query_labels']
        candidate_labels = inputs['candidate_labels']
        predictions, query_embeddings, candidate_embeddings, query_label_predictions, candidate_label_predictions = model(
            inputs, training=False)
        loss = cal_loss(query_labels, candidate_labels, score, predictions, query_label_predictions,
                        candidate_label_predictions) * score.shape[-1]  # convert mean back to sum
        val_recorder.record(loss, score, predictions)
        return score, predictions

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
                vid_pred = []
                label = []
                for val_batch in val_dataset:
                    score, predictions = val_step(val_batch)
                    vid_pred.extend(predictions.numpy().tolist())
                    label.extend(score.numpy().tolist())

                # 8. test spearman correlation
                spearmanr = scipy.stats.spearmanr(vid_pred, label).correlation
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearmanr > 0.6:
                    checkpoint_manager.save(checkpoint_number=step)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
