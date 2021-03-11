from platform import platform
import socket
from os.path import join


MACHINE_NAME = socket.gethostname()
if platform().startswith('Windows'):
    PLATFORM = 'Windows'
    DATA_DIR = 'd:/data'
    OUTPUT_DIR = 'd:/data/output'
elif MACHINE_NAME.startswith('rgcpu5'):
    PLATFORM = 'Linux'
    DATA_DIR = '/data/hldai/data'
    OUTPUT_DIR = '/data/hldai/data/'
else:
    PLATFORM = 'Linux'
    DATA_DIR = '/home/data/hldai/'
    OUTPUT_DIR = '/data/hldai/'

LOG_DIR = join(OUTPUT_DIR, 'log')
