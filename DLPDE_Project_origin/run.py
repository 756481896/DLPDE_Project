import tensorflow as tf
from data_loader.data_loader import *
from models.DLPDE_Model import *
from trainers.Trainer import *
from utils.config import *
from utils.dirs import *
from utils.utils import get_args
from utils.logger import Logger
import pickle
import math
import os
import random
import sys
import time
import random
import json
import kenlm 
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
import pdb
tf_config = tf.ConfigProto(allow_soft_placement = True)
tf_config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create tensorflow session
    tf.reset_default_graph()
    sess = tf.Session(config = tf_config)
    # create instance of the model you want
    model = DLPDE_Model(config)
    model.load(sess)
    # create your data generator
    train_data,test_data, input_train_extra = creat_dataset(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = Trainer(sess, model, train_data, test_data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
