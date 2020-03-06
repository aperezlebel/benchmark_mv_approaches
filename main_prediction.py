import logging
import sys
from prediction.main import main

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

print_file = open('prediction_print.log', 'w')
sys.stdout = print_file

logging.basicConfig(
    filename='prediction.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger.info('Started prediction')
main()
logger.info('Ended prediction')

print_file.close()
