# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(handler)

RANDOM_SEED = 500
DATASET_CLASSES = {"cifar10":10, "mnist":10, "cifar100":100}
