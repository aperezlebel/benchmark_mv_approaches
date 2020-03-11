"""Main script. Configure logger, load .env and run main."""
import logging
import sys
import os
import main
from dotenv import load_dotenv


################ CONFIGURE LOGGER ################
logs_folder = 'logs/'

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

def get_and_increase_count():
    count_filepath = logs_folder+'count.txt'

    if not os.path.exists(count_filepath):
        count = 0
    else:
        with open(count_filepath, 'r') as file:
            c = file.read()
            if c == '':
                count = 0
            else:
                count = int(c) + 1

    # Dump new count
    os.makedirs(logs_folder, exist_ok=True)
    with open(count_filepath, 'w') as file:
        file.write(str(count))

    return count


count = get_and_increase_count()


def get_log_filepath(filename):
    return f'{logs_folder}{count}_{filename}'


# print_file = open(get_log_filepath('prediction_print.log'), 'w')
# sys.stdout = print_file

logging.basicConfig(
    filename=get_log_filepath('prediction.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

################ LOAD ENV ################
load_dotenv()

################ RUN ################
logger.info('Started run')
main.run(sys.argv)
logger.info('Ended run')

# print_file.close()
