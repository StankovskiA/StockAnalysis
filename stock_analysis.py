from constants import DATA_DIR, ORIGINAL_FILE
from EDA import data_exploration
import pandas as pd
import logging
import os
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
def main():
    df = pd.read_csv(os.path.join(DATA_DIR, ORIGINAL_FILE))

    logger.info('Data Exploration')
    data_exploration(df)
    
if __name__ == '__main__':
    main()