import os
import numpy as np
import sys
sys.path.append(os.path.join(".", "util"))
from NewsProcessor import NewsProcessor

# PARAMETERS
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

def main(
    path_news_code: str = PATH_NEWS_CODE,
    path_news_train: str = PATH_NEWS_TRAIN,
    path_output_pairs: str = PATH_OUTPUT_PAIRS,
    number_of_pairs: int = NUMBER_OF_PAIRS,
    use_cosine: bool = USE_COSINE,
    use_jaccard: bool = USE_JACCARD,
    min_appearances: int = MIN_APPEARANCES,
    max_appearances: int = MAX_APPEARANCES,
    similarity_threshold: float = HEADLINE_SIMILARITY_THRESHOLD,
) -> None:
    """
    Create random pairs of news articles and save them to a CSV file in the output directory.

    Args:
        path_news_code (str): path to news_code.csv
        path_news_train (str): path to news directory
        path_output_pairs (str): path to output directory
        number_of_pairs (int): number of pairs to create
        use_cosine (bool): whether to use cosine similarity
        use_jaccard (bool): whether to use jaccard similarity
        min_appearances (int): minimum number of appearances of a word in the corpus
        max_appearances (int): maximum number of appearances of a word in the corpus
        similarity_threshold (float): threshold for headline similarity
    """

    print(f"\nStarting the process of creating pairs of news articles...\n")

    # initialize the NewsProcessor 
    processor = NewsProcessor()

    # preprocessing the news articles with the given parameters 
    processor.process(path_news_code, path_news_train, min_appearances, max_appearances, debug=False)

    # create random pairs of news articles 
    if not (use_cosine or use_jaccard):
        # if both use_cosine and use_jaccard are False, raise an error
        raise ValueError("At least one of use_jaccard and use_cosine must be True !")
    else:
        # create pairs of news articles and save them to a CSV file
        processor.generate_pairs(
            num_pairs=number_of_pairs, 
            filepath=os.path.join(path_output_pairs, f'pairs_{number_of_pairs}.csv'),
            use_cosine=use_cosine, 
            use_jaccard=use_jaccard, 
            headline_similarity_threshold=similarity_threshold
        )


if __name__ == "__main__":
    main()