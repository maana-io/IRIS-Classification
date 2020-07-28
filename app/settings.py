import os
from dotenv import load_dotenv
import logging


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
load_dotenv(verbose=True, dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

LOG_LEVEL = logging.DEBUG

NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIR')
W2VEC_MODEL_DIR = os.getenv('W2VEC_MODEL_DIR')

CLASSIFICATION_DATA_DIR = os.getenv('CLASSIFICATION_DATA_DIR')
