import os
import math
import numpy as np
import pdb

def creat_data(sample_num_0,sample_num_1,sample_num_2 = 0, extra_sample_num = 0):
    """
    根据真实值生成训练数据
    sample_num_0 : x = -pi , t = random s时u的数据 以及 x= pi , t = random s时u的数据
    sample_num_1 : x = random , t = 0 时 u 的 数据。
    sample_num_2 : x = random , t = random u 的数据， 不在边界条件上的点，样本数可以为0
    extra_sample_num : 不用已知的u的值，只有输入的x,t ，用于对f的结果进行额外的训练。此时不需要输出值。
    """
    t00 = np.random.uniform(size = sample_num_0)
    x00 = np.ones(sample_num_0) * (-np.pi)
    u00 = exact_solution(x00,t00)
    #x = -pi , t = random s时u的真实值
    
    t01 = np.random.uniform(size = sample_num_0)
    x01 = np.ones(sample_num_0) * (np.pi)
    u01 = exact_solution(x01,t01)
    #x = pi , t = random s时u的真实值
    
    x1 = np.random.uniform(low = - math.pi , high = math.pi , size = sample_num_1)
    t1 = np.zeros(sample_num_1)
    u1 = exact_solution(x1,t1)
    # x = random , t = 0 时 u 的 真实值。
    
    x2 = np.random.uniform(low = - math.pi , high = math.pi , size = sample_num_2)
    t2 = np.random.uniform(size = sample_num_2)
    u2 = exact_solution(x2,t2)
    # x = random , t = random u 的真实值， 不在边界条件上的点，样本数可以为0
    
    input_x = np.concatenate([x00,x01,x1,x2])
    input_t = np.concatenate([t00,t01,t1,t2])
    input_data = np.stack([input_x,input_t],axis =0)
    
    output_data = np.concatenate([u00,u01,u1,u2])
    output_data_noise = include_noise(output_data)
    output_data_noise = np.reshape(output_data_noise,[-1,1])
    #构建输入输出的训练数据，f为0 ，不引入误差
    
    x_extra = np.random.uniform(low = - math.pi , high = math.pi , size = extra_sample_num)
    t_extra = np.random.uniform(size = extra_sample_num)
    input_extra = np.stack([x_extra,t_extra],axis =0)
    return input_data.T,output_data_noise,input_extra.T
    #shape: (sample_num,2),(sample_num,1),(sample_num,2)

def creat_dataset(config):
    """制造训练集和测试集"""
    input_train_data,output_train_data,input_train_extra = creat_data(config.train_sample_num_0,config.train_sample_num_1,config.train_sample_num_2, config.extra_sample_num)
    input_test_data,output_test_data,_ = creat_data(config.test_sample_num_0,config.test_sample_num_1,config.test_sample_num_2, extra_sample_num = 0)
    index = np.random.choice(len(input_train_data),size = len(input_train_data),replace=False)
    input_train_data,output_train_data = input_train_data[index], output_train_data[index]
    #打乱训练集
    # return input_train_data,output_train_data,input_train_extra,input_test_data,output_test_data
    train_data = [input_train_data,output_train_data]
    test_data = [input_test_data,output_test_data]
    return train_data,test_data, input_train_extra
    
def exact_solution(x,t):
    """输入x,t，返回准确解"""
    return np.exp(-t)*np.sin(x)

def include_noise(x,noise_level = 1e-2):
    """引入正态分布的噪音"""
    if len(np.shape(x)) == 1:
        y = x + np.random.randn(np.shape(x)[0]) * noise_level
    elif len(np.shape(x)) == 2:
        y = x + np.random.randn(np.shape(x)[0],np.shape(x)[1]) * noise_level
    else:
        raise IOError("输入维度错误")
    return y

def pair_iter(input_data,output_data,batch_size):
    """
    按batch生成训练数据
    """
    batches = (len(input_data) + batch_size - 1)//batch_size
    for i in range(batches-1):
        input_data_batch = input_data[i*batch_size:(i+1)*batch_size]
        x_batch = np.reshape(input_data_batch[:,0],[-1,1])
        #shape: [batch_size,1]
        t_batch = np.reshape(input_data_batch[:,1],[-1,1])
        output_data_batch = output_data[i*batch_size:(i+1)*batch_size]
        output_data_batch = np.reshape(output_data_batch,[-1,1])
        yield (x_batch,t_batch,output_data_batch)
        # yield (x_batch,t_batch,u_batch)