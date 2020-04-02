"""Main script. Configure logger, load .env and run main."""
import logging
import sys
import os
import random
import string


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
    slug = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    return f'{logs_folder}{count}_{slug}_{filename}'


# print_file = open(get_log_filepath('prediction_print.log'), 'w')
# sys.stdout = print_file

log_filepath = get_log_filepath('prediction.log')

logging.basicConfig(
    filename=log_filepath,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
import main  # Delayed import because logger need to be inititialized


################ RUN ################
logger.info('Started run')
print(f'Dumping logs into {log_filepath}')
main.run(sys.argv)
logger.info('Ended run')

# print_file.close()
