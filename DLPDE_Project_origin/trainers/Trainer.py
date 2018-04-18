from base.base_train import BaseTrain
from tqdm import tqdm
# from models.test_model import *
import numpy as np
from data_loader.data_loader import *
import pdb
import time
import sys
import logging
logging.basicConfig(level=logging.INFO)

class Trainer(BaseTrain):
    def __init__(self, sess, model, train_data, test_data, config, logger):
        """data:[input_data,output_data]"""
        super(Trainer, self).__init__(sess, model, train_data, test_data, config, logger)
    
    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop ever the number of iteration in the config and call teh train step
       -add any summaries you want using the summary
        """
        previous_losses = []
        exp_loss = None
        for x_batch,t_batch,output_data_batch in pair_iter(self.train_data[0],self.train_data[1],self.config.batch_size):
            tic = time.time()
            u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate = self.train_step(self.sess,x_batch,t_batch,output_data_batch)
            toc = time.time()
            iter_time = toc-tic
            step = self.sess.run(self.model.global_step_tensor)
            epoch = self.model.cur_epoch_tensor.eval(self.sess)
            total_sample = step*self.config.batch_size

            if type(exp_loss) != np.ndarray:
                exp_loss_0 = loss_0
                exp_loss_sum = loss_sum
            else:
                exp_loss_0 = 0.99*exp_loss_0 + 0.01*loss_0
                exp_loss_sum = 0.99*exp_loss_sum + 0.01*loss_sum
            #损失和准确率的滑动平均
            if step %10 ==0:
                logging.info("time %f, epoch %d, step %d,sample %d, loss_sum %f, loss_0 %f, loss_1 %f exp_loss_sum %f, exp_loss_0 %f, mean u %f, mean v %f, mean f %f, learning_rate %f, grad_norm %f, param norm %f" % (iter_time,epoch,step,total_sample,loss_sum,loss_0,loss_1,exp_loss_sum,exp_loss_0,np.mean(u),np.mean(v),np.mean(f),learning_rate,gradient_norm,param_norm))
                # self.writer.add_summary(summary,step)
                summaries_dict = {}
                summaries_dict['loss_sum'] = loss_sum
                summaries_dict['exp_loss_sum'] = exp_loss_sum
                summaries_dict['loss_0'] = loss_0
                summaries_dict['exp_loss_0'] = exp_loss_0
                summaries_dict['loss_1'] = loss_1
                summaries_dict['mean_u'] = np.mean(u)
                summaries_dict['mean_v'] = np.mean(v)
                summaries_dict['mean_f'] = np.mean(f)
                summaries_dict['grad_norm'] = gradient_norm
                self.logger.summarize(step, summaries_dict=summaries_dict)
        valid_loss_sum,valid_loss_0,valid_loss_1 = self.validate()
        
        if len(previous_losses) > 2 and valid_loss_sum > previous_losses[-1]:
            #当损失与上一个epoch相比反而增加了，将参数值返回到最佳的epoch时的参数，并且降低学习率
            logging.info("Annealing learning rate by %f" % self.config.learning_rate_decay_factor)
            self.sess.run(self.model.learning_rate_decay_op)
            # model.saver.restore(sess, self.config.checkpoint_name + ("-%d" % best_epoch))
            #学习率递减
        else:
            previous_losses.append(valid_loss_sum)
            best_epoch = epoch
            print('saving...')
            self.model.saver.save(self.sess,self.config.checkpoint_name,global_step = epoch)
            print('saved.')
        sys.stdout.flush()

    def train_step(self,sess,x_batch,t_batch,output_data_batch):
        """
        训练过程
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        input_feed = {}
        input_feed[self.model.x] = x_batch
        input_feed[self.model.t] = t_batch
        input_feed[self.model.output_placeholder] = output_data_batch
        output_feed = [self.model.updates,self.model.u, self.model.v, self.model.f, self.model.loss_0, self.model.loss_1,self.model.loss_sum, self.model.gradient_norm,self.model.param_norm,self.model.learning_rate]
        _,u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate= sess.run(output_feed,input_feed)
        sess.run(self.model.increment_global_step_tensor)
        #步数计算器
        return u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate
    
    def test_step(self,sess,x_batch,t_batch,output_data_batch):
        """
        测试过程
        implement the logic of the test step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        input_feed = {}
        input_feed[self.model.x] = x_batch
        input_feed[self.model.t] = t_batch
        input_feed[self.model.output_placeholder] = output_data_batch
        output_feed = [self.model.u, self.model.v, self.model.f, self.model.loss_0, self.model.loss_1,self.model.loss_sum, self.model.gradient_norm,self.model.param_norm,self.model.learning_rate]
        u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate= sess.run(output_feed,input_feed)
        return u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate

    def validate(self):
        input_data_valid, output_data_valid = self.test_data[0],self.test_data[1]
        x_valid = np.reshape(input_data_valid[:,0],[-1,1])
        t_valid = np.reshape(input_data_valid[:,1],[-1,1])
        epoch = self.model.cur_epoch_tensor.eval(self.sess)
        tic = time.time()
        u, v, f, loss_0, loss_1,loss_sum, gradient_norm,param_norm,learning_rate = self.test_step(self.sess,x_valid,t_valid,output_data_valid)
        toc = time.time()
        iter_time = toc-tic
        logging.info("TEST:loss_sum %f, loss_0 %f, loss_1 %f,mean u %f, mean v %f, mean f %f, learning_rate %f, grad_norm %f, param norm %f" % (loss_sum,loss_0,loss_1,np.mean(u),np.mean(v),np.mean(f),learning_rate,gradient_norm,param_norm))
            # self.writer.add_summary(summary,step)
        summaries_dict = {}
        summaries_dict['valid_loss_sum'] = loss_sum
        summaries_dict['valid_loss_0'] = loss_0
        summaries_dict['valid_loss_1'] = loss_1
        summaries_dict['valid_mean_u'] = np.mean(u)
        summaries_dict['valid_mean_v'] = np.mean(v)
        summaries_dict['valid_mean_f'] = np.mean(f)
        summaries_dict['valid_grad_norm'] = gradient_norm
        self.logger.summarize(epoch, summaries_dict=summaries_dict)
        return loss_sum,loss_0,loss_1