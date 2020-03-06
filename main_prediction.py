import logging
import sys
from prediction.main import main

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

sys.stdout = open('prediction_print.log', 'w')

logging.basicConfig(filename='prediction.log', filemode='w', level=logging.INFO)#, format='[%(asctime)s] %(message)s')
logger.info('Started prediction')
main()
logger.info('Ended prediction')
