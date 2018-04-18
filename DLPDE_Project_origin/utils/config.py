import json
from bunch import Bunch
import os
from utils.dirs import *

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config, config_dict

def save_config_to_json(dict_ob,save_name):
    """
    把dict文件保存为json格式,保存在configs目录下
    """
    json_file = 'configs/'+save_name
    with open(json_file,'w') as config_file:
        json.dump(dict_ob,config_file)
    

def process_config(jsonfile):
    config, config_dict = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join("/data2/pengqiu/experiments", config.exp_name, "summary")
    config.checkpoint_dir = os.path.join("/data2/pengqiu/experiments", config.exp_name, "checkpoint")
    config.checkpoint_name = os.path.join(config.checkpoint_dir, config.save_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])
    #保存当前使用的config
    config_path = os.path.join("/data2/pengqiu/experiments", config.exp_name) +"/config.json"
    with open(config_path,'w') as config_file:
        json.dump(config_dict, config_file)
    # config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "tensorboard")
    return config
