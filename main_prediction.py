import logging
from prediction.main import main

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

logging.basicConfig(filename='prediction.log', filemode='w', level=logging.INFO)#, format='[%(asctime)s] %(message)s')
logger.info('Started prediction')
main()
logger.info('Ended prediction')
