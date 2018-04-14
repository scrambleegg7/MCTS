#

import logging

def get_module_logger(modname):
    logger = logging.getLogger(modname)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    #handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
