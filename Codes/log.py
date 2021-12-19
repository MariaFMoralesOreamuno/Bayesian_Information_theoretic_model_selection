from config import *

"""
Author: Maria Fernanda Morales Oreamuno 

File generates logging information 
"""

# define name and location of logfile
log_file = "./logfile.log"

# generate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create formatter:
formatter = logging.Formatter('%(asctime)s:%(levelname)s:[%(filename)s:%(lineno)s] - %(message)s')
formatter_print = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

# to add a handler to save logs to a file:
file_handler = logging.FileHandler('logfile.log')  # which file handles the loggings
file_handler.setFormatter(formatter)

# to print logging:
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter_print)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
