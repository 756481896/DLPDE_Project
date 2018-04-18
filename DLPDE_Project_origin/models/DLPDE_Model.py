from base.base_model import BaseModel
import tensorflow as tf
import tensorflow as tf
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
import pdb
logging.basicConfig(level=logging.INFO)
class DLPDE_Model(BaseModel):
    def __init__(self, config):
        super(DLPDE_Model, self).__init__(config)
        self.build_model()
        self.init_saver()
        
    def build_model(self):
        self.learning_rate_decay_factor = self.config.learning_rate_decay_factor
        self.learning_rate = tf.Variable(float(self.config.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
        self.optimizer = self.config.optimizer
        #模型创建
        with tf.variable_scope('forward') as scope:
            # self.input_placeholder = tf.placeholder(shape=[None,2],dtype=tf.float32)
            # self.x = self.input_placeholder[:,0]
            # self.t = self.input_placeholder[:,1]
            self.x = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.t = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.input = tf.concat([self.x,self.t],1)
            #输入，为[x,t],None表示batch_size
            self.u = self.neural_net(self.input,scope,num_layers = self.config.num_layers, neural_size = self.config.neural_size)
            self.v = tf.Variable(tf.random_normal(shape=[1,1]), trainable=True, dtype=tf.float32)
            #未知参数
            self.u_t = tf.gradients(self.u,self.t)[0]
            self.u_x = tf.gradients(self.u,self.t)[0]
            self.u_xx = tf.gradients(self.u_x,self.x)[0]
            self.f = self.u_t-self.v*self.u_xx
            
        self.output_placeholder = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.loss_0 = tf.losses.mean_squared_error(self.output_placeholder, self.u)
        #u处的损失
        zero = tf.fill(tf.shape(self.f), 0.0)
        #全为0的constant
        self.loss_1 = tf.losses.mean_squared_error(self.f, zero)
        self.loss_sum = self.loss_0+self.loss_1

        self.setup_update()
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        
    def neural_net(self,input,scope,num_layers = 4,neural_size = 20, output_size = 1):
        """neural_size:中间节点数，output_size:输出节点数"""
        layer_tmp = input
        for i in range(num_layers-1):
            layer_tmp = tf.contrib.layers.fully_connected(layer_tmp,neural_size)
        u = tf.contrib.layers.fully_connected(layer_tmp,output_size)
        return u
    def setup_update(self):
        """计算梯度并且update"""
        params = tf.trainable_variables()
        opt = self.get_optimizer(self.optimizer)(self.learning_rate)
        if self.config.extra_train == 'True':
            #用于对f进行额外的训练
            gradients = tf.gradients(self.loss_1, params)
        else:
            gradients = tf.gradients(self.loss_sum, params)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(zip(gradients, params), global_step=self.global_step_tensor)
        
    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)