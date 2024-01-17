import numpy as np
import torch
import sys
import os.path
sys.path.append(os.path.join(".", "script")) 
from create_pairs import main as create_pairs 
from train_model import main as train_model
from model_evaluation import main as model_evaluation

# CONFIGURATION
# fix random seed for reproducibility for comparison
SEED: int = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# PARAMETERS - CHOOSE SCRIPTS TO RUN
CREATE_PAIRS: bool = True # whether to create pairs of news (if False, need to have pairs_{}.csv in the output folder)
TRAIN_MODEL: bool = True # whether to train the model (if False, need to have the necessary model files in the model folder)
MODEL_EVALUATION: bool = True # whether to test the model by classifying news and comparing the results with baseline 

# PARAMETERS - CREATE PAIRS
# paths
PATH_NEWS_CODE: str = os.path.join('data', 'news_code.csv') # path to news_code.csv
PATH_NEWS_TRAIN: str = os.path.join('data', 'news_train') # path to news files for training
PATH_OUTPUT_PAIRS: str = os.path.join('data', 'pairs') # path to output directory for pairs of news
# parameters of the script
NUMBER_OF_PAIRS : int = 200 # number of pairs to create
USE_COSINE : bool = True # whether to compute cosine similarity between headlines
USE_JACCARD : bool = True # whether to compute jaccard similarity between headlines
MIN_APPEARANCES : int = 100 # minimum number of appearances of a word in the corpus (see NewsProcessor.py for more info)
MAX_APPEARANCES : int = np.inf # maximum number of appearances of a word in the corpus (see NewsProcessor.py for more info)
HEADLINE_SIMILARITY_THRESHOLD : float = 0.75 # threshold for maximum headline similarity to create a pair of news

# PARAMETERS - TRAIN MODEL
# paths
PATH_DATA_PAIRS: str = PATH_OUTPUT_PAIRS # path to the pairs data for training (pairs_{}.csv)
PATH_MODEL_SAVE: str = os.path.join('model') # output path to save the model
# parameters of the script
DATA_SIZE_PAIRS: int = NUMBER_OF_PAIRS # size of the dataset to use for training
MODEL_NAME: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' # pre-trained Sentence-Transformer model to use 
LABEL: str = "similarity_cosine" # choose label to use, either 'similarity_cosine', 'similarity_jaccard'
# hyperparameters to tune
EPOCHS: int = 8 # number of epochs to train the model 
SCHEDULER: str = 'warmupLinear' # scheduler to change learning rate, either 'warmupConstant', 'warmupLinear', 'warmupCosine'

# PARAMETERS - EVALUATE MODEL (CLASSIFICATION)
# paths
PATH_NEWS_CODE: str = os.path.join('data', 'news_code.csv') # path to news code file
PATH_NEWS_TEST: str = os.path.join('data', 'news_test') # path to news data for testing
PATH_OUTPUT_SCORES: str = os.path.join('scores', 'classification_scores.csv') # path to output file of scores for classification
PATH_MODEL: str = PATH_MODEL_SAVE # path to load the local SentenceTransformer model
# parameters of the script
DATA_SIZE_EVALUATION: int = 100 # number of news to use for evaluation
N_SUBJECTS: int = -1 # number of subjects per news to keep (-1 means all subjects)
BASELINE_ST_MODEL: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' # pre-trained Sentence-Transformer model to use for baseline


def main() -> None:
    """
    Main function to run the whole process of creating pairs of news, training the model and testing it. 
    Set the parameters above to True or False to choose which processes to run.
    """

    if CREATE_PAIRS :
        print("\nCREATE PAIRS")
        print("=================\n")
        create_pairs(
            path_news_code=PATH_NEWS_CODE, 
            path_news_train=PATH_NEWS_TRAIN, 
            path_output_pairs=PATH_OUTPUT_PAIRS, 
            number_of_pairs=NUMBER_OF_PAIRS,
            use_cosine=USE_COSINE, 
            use_jaccard=USE_JACCARD,
            min_appearances=MIN_APPEARANCES,
            max_appearances=MAX_APPEARANCES,
            similarity_threshold=HEADLINE_SIMILARITY_THRESHOLD
        )
        print("\n-----------------\n\n")

    if TRAIN_MODEL :
        print("\nTRAIN MODEL")
        print("=================\n")
        train_model(
            path_data_pairs=PATH_DATA_PAIRS,
            path_model_save=PATH_MODEL_SAVE,
            data_size=DATA_SIZE_PAIRS,
            model_name=MODEL_NAME,
            label=LABEL,
            epochs=EPOCHS,
            scheduler=SCHEDULER,
        )
        print("\n-----------------\n\n")
        
    if MODEL_EVALUATION :
        print("\nMODEL EVALUATION")
        print("=================\n")
        model_evaluation(
            path_news_code=PATH_NEWS_CODE,
            path_news_test=PATH_NEWS_TEST,
            path_output_scores=PATH_OUTPUT_SCORES,
            path_model=PATH_MODEL,
            data_size=DATA_SIZE_EVALUATION,
            n_subjects=N_SUBJECTS,
            baseline_st_model=BASELINE_ST_MODEL
        )
        print("\n-----------------\n\n")

    print("\nAll the processes have been run.\n")


if __name__ == '__main__':
    main()