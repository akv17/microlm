import logging

import yaml


def create_logger():
    logger = logging.getLogger('microlm')
    logger.setLevel('INFO')
    logging.basicConfig()
    return logger


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
