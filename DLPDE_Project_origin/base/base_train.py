import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np


class BaseTrain:
    def __init__(self, sess, model, train_data, test_data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        # self.setup_writer()
    def setup_writer(self):
        """设置tensorboard的writer"""
        tensorboard_dir = self.config.tensorboard_dir
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.writer = tf.summary.FileWriter(tensorboard_dir)
        self.writer.add_graph(sess.graph)
    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the sammary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
