import numpy as np
import time 
import os

def top_k(input,n = 3,axis = 1):
    """
    Returns:
      values: The `k` largest elements along axis slice.
      indices: The indices of `values` within axis of `input`.
    """
    input = input.copy()
    if axis == 0:
        n_indice = input.argsort(axis=axis)[-n:,:][::-1,:]
        input.sort(axis=axis)
        n_value = input[-n:,:][::-1,:]
        return n_indice,n_value
    
    n_indice = input.argsort(axis=axis)[:,-n:][:,::-1]
    input.sort(axis=axis)
    n_value = input[:,-n:][:,::-1]
    return n_indice,n_value

def rokid_cut_file(input_file_name,output_file_name):
    """用rokid 分词对input file 进行分词,输出到output_file"""
    cd = 'cd /data2/pengqiu/nlp2.0/engine/src/unittest/;'
    wordsegment =  './WordSegmentTest < '
    command = cd + wordsegment + input_file_name +' > ' + output_file_name
    # res = os.popen('cd /data2/pengqiu/nlp2.0/engine/src/unittest/; ./WordSegmentTest < /data2/pengqiu/LM_data/rokid_test_data.txt > /data2/pengqiu/LM_data/NEWS/rokid_test_data_cut.txt')
    res = os.popen(command)
    return res

def RNNLM_model_predict(model,input_text,Print=True):
    """输入一句话，输出每个字的概率"""
    vocab_to_int = model.vocab_to_int
    one_token = []
    one_token = [2] + [vocab_to_int[i] for i in input_text] +[3]
    one_token = np.reshape(one_token,[-1,1])
    
    
    
    
    return 0