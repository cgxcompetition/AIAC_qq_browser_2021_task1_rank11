import logging

import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.meansquarederror = tf.keras.metrics.MeanSquaredError()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, meansquarederror: {:.4f} '

    def record(self, losses, labels, predictions):
        self.loss.update_state(losses)
        self.meansquarederror.update_state(labels, predictions)

    def reset(self):
        self.loss.reset_states()
        self.meansquarederror.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        meansquarederror = self.meansquarederror.result().numpy()
        return [loss, meansquarederror]

    def score(self):
        return self._results()[-1].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss, meansquarederror = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss, meansquarederror) + suffix)
