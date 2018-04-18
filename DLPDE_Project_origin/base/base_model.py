import tensorflow as tf
import os


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step_tensor)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir))
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        else:
            print("Load fail!")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self):
        # self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def get_optimizer(self,opt):
        """设置optimizer: adam 或 rmsprop"""
        if opt == "adam":
            optfn = tf.train.AdamOptimizer
        elif opt == 'rmsprop':
            optfn = tf.train.RMSPropOptimizer
        else:
            optfn = tf.train.GradientDescentOptimizer
        return optfn