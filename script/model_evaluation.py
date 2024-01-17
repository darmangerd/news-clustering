import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append(os.path.join(".", "util"))
from NewsProcessor import NewsProcessor

# CONFIGURATION
# avoid unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# fix random seed for reproducibility and comparison
SEED: int = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# PARAMETERS
# paths
PATH_NEWS_CODE: str = os.path.join('data', 'news_code.csv') # path to news code file
PATH_NEWS_TEST: str = os.path.join('data', 'news_test') # path to news data for testing
PATH_OUTPUT_SCORES: str = os.path.join('scores', 'classification_scores.csv') # path to output file of scores for classification
PATH_MODEL: str = os.path.join('model') # path to load the local SentenceTransformer model
# parameters of the script
DATA_SIZE: int = 200 # number of news to use (-1 means all news)
N_SUBJECTS: int = 5 # number of subjects per news to keep (-1 means all subjects)
BASELINE_ST_MODEL: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' # pre-trained Sentence-Transformer model to use for baseline


def encode(news: pd.DataFrame, model: SentenceTransformer) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode the news using the provided model and return the encoded news and subjects.

    Args:
        news (pandas.DataFrame): news dataframe
        model (SentenceTransformer): model to use for encoding

    Returns:
        tuple: X, y (encoded news and subjects)
    """
    # get news 'headline' in a list
    corpus = news['headline'].tolist()
    # encode the news
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    # add encoded news to news dataframe
    news['embedding'] = corpus_embeddings.tolist()

    # convert lists in 'embedding' and 'subjects_encoded' columns to numpy arrays (for compatibility wth multioutput classifier)
    news.loc[:, 'embedding'] = news['embedding'].apply(lambda x: np.array(x))
    news.loc[:, 'subjects_encoded'] = news['subjects_encoded'].apply(lambda x: np.array(x))
    X = np.stack(news['embedding'].values)
    y = np.stack(news['subjects_encoded'].values)

    return X, y


def metrics_compute(y_test: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute the accuracy, recall, precision, and F1 score of the classifier on the test set.

    Args:
        y_test (numpy.ndarray): actual subjects
        y_pred (numpy.ndarray): predicted subjects

    Returns:
        tuple: accuracy, recall, precision, f1 (metrics)
    """
    # measure the accuracy of the classifier
    # accuracy in this multi-label setting is the subset accuracy 
    # it means that the predicted subjects must match exactly the actual subjects
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The accuracy of the classifier on the test set is: {accuracy*100:.2f}%.")

    # measure the recall of the classifier
    # average='samples' means that we compute the recall for each sample, then average the results
    recall = recall_score(y_test, y_pred, average='samples', zero_division=0)
    print(f"The recall of the classifier on the test set is: {recall*100:.2f}%.")

    # measure the precision of the classifier
    precision = precision_score(y_test, y_pred, average='samples', zero_division=0)
    print(f"The precision of the classifier on the test set is: {precision*100:.2f}%.")

    # measure the F1 score of the classifier
    f1 = f1_score(y_test, y_pred, average='samples', zero_division=0)
    print(f"The F1 score of the classifier on the test set is: {f1*100:.2f}%.")

    return accuracy, recall, precision, f1


def classifier_evaluation(news: pd.DataFrame, X: np.ndarray = None, y: np.ndarray = None, model: SentenceTransformer = None) -> tuple[float, float, float, float]:
    """
    Train a multioutput classifier on the news data and evaluate it.

    In detail: train a multioutput classifier on the training set (X: news embeddings by the model, y: encoded subjects)
    , evaluate the classifier on the test set by comparing the predicted subjects to the actual subjects

    Args:
        news (pandas.DataFrame): news dataframe
        X (numpy.ndarray): news embeddings by the model
        y (numpy.ndarray): encoded subjects
        model (SentenceTransformer): model to use for encoding

    Returns:
        tuple: accuracy, recall, precision, f1 (metrics)
    """
    # encode the news using the provided model (else use the provided X and y)
    if model is not None and (X is None or y is None):
        X, y = encode(news.copy(), model)

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # train a multioutput classifier
    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    multi_target_forest.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = multi_target_forest.predict(X_test)

    # compute metrics and return them
    return metrics_compute(y_test, y_pred)    


def main(
    path_news_code: str = PATH_NEWS_CODE,
    path_news_test: str = PATH_NEWS_TEST,
    path_output_scores: str = PATH_OUTPUT_SCORES,
    path_model: str = PATH_MODEL,
    data_size: int = DATA_SIZE,
    n_subjects: int = N_SUBJECTS,
    baseline_st_model: str = BASELINE_ST_MODEL
) -> None:  
    """
    Model evaluation script.
    
    Args:
        path_news_code (str): path to news code file
        path_news_test (str): path to news test folder
        path_output_scores (str): path to output scores file
        path_model (str): path to load the local model
        data_size (int): number of news to use
        n_subjects (int): number of subjects per news to keep (-1 means all subjects)
        baseline_st_model (str): pre-trained model to use for baseline
    """
    print("\nStarting model evaluation process...\n")
    
    # load news data using NewsProcessor class
    processor = NewsProcessor()
    # process news data
    processor.process(path_news_code, path_news_test)
    # filter top subjects per news to reduce the number of total subjects
    if n_subjects != -1:
        processor.select_top_subjects_per_news(n=n_subjects)
    # get news data
    news = processor.get_news()
    # add encoded subjects to news dataframe to use for classification
    news['subjects_encoded'] = processor.encode_subjects().tolist()
    # select a random sample of news if data_size is not -1
    if data_size != -1:
        # get random sample of news based on data_size
        news = news.sample(n=data_size, random_state=SEED)
    else:
        data_size = len(news)

    # load model fine-tuned on the news data
    transformer_model = SentenceTransformer(path_model)

    # BASELINE 1 : model without fine-tuning (parameters should be the same as the fine-tuned model)
    # max_seq_length : max number of words in a sentence
    word_embedding_model = models.Transformer(baseline_st_model, max_seq_length=256) 
     # pooling model for sentence embedding
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        # dimension of the input vector
        in_features=pooling_model.get_sentence_embedding_dimension(), 
         # dimension of the output vector
        out_features=256,
        # activation function (tanh, relu, ...)
        activation_function=nn.Tanh()) 
    model_baseline_1 = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model]) 

    # store sentence-transformers models in a list
    sentence_transformers_models = [(transformer_model, 'fine-tuned'), (model_baseline_1, 'without fine-tuning')]

    # initialize dataframe to store model scores
    model_scores = pd.DataFrame(columns=['model', 'accuracy', 'recall', 'precision', 'f1'])

    # evaluation of the models from the list
    for model, name in sentence_transformers_models:
        print(f"\n\nEvaluating the {name} model...\n")

        # train and evaluate the model
        accuracy, recall, precision, f1 = classifier_evaluation(news.copy(), model=model)

        # add scores to dataframe
        model_scores = pd.concat([model_scores, pd.DataFrame([[name, accuracy, recall, precision, f1]], columns=['model', 'accuracy', 'recall', 'precision', 'f1'])])
        print(f"\nFinished processing {name}.")

    # BASELINE 2 : random prediction based on subject frequencies
    # reference : https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    print("\n\nEvaluating the random prediction based on subject frequencies...\n")

    # encode and split the data (we have to use the transformer model to encode the news as the baseline model doesn't have an encode method)
    # the model used here doesn't matter as we only need the encoded news
    X, y = encode(news.copy(), transformer_model)
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # create a dummy classifier that predicts according to the subject frequencies.
    # DummyClassifier is a simple rule-based classifier provided by sklearn, which is useful as a simple baseline to compare with other classifiers.
    # the "stratified" strategy generates predictions by respecting the class distribution of the training set.
    # this means that the probability of predicting each class is proportional to its frequency in the training set.
    dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
    # fit the classifier on the training set
    dummy_clf.fit(X_train, y_train)

    # make predictions on the test set using the trained dummy classifier
    y_pred = dummy_clf.predict(X_test)

    # compute metrics 
    accuracy, recall, precision, f1 = metrics_compute(y_test, y_pred)

    # add scores from this baseline to dataframe
    model_scores = pd.concat([model_scores, pd.DataFrame([['random frequency-based', accuracy, recall, precision, f1]], columns=['model', 'accuracy', 'recall', 'precision', 'f1'])])
    print(f"\nFinished processing random frequency-based.\n")

    # BASELINE 3 : using tf-idf encoding (text is considered as bag of words without order)
    # reference : https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    print("\nEvaluating the TF-IDF encoding...\n")

    # the TfidfVectorizer transforms text data into numerical features
    # it does this by turning the text into a "bag of words" representation, where the order of words doesn't matter
    # and assigning a weight to each word based on its frequency in the document and in the corpus as a whole (tf-idf)

    # max number of features to extract from the text (limit the size of the vocabulary for memory management)
    vectorizer = TfidfVectorizer(max_features=650)
    X = vectorizer.fit_transform(news['headline'].tolist()).toarray()
    # get the targets (subjects) for each news, stored in 'subjects_encoded' column
    y = np.stack(news['subjects_encoded'].values)

    # evaluate the model
    accuracy, recall, precision, f1 = classifier_evaluation(news.copy(), X, y)

    # add scores from this baseline to dataframe
    model_scores = pd.concat([model_scores, pd.DataFrame([['tf-idf', accuracy, recall, precision, f1]], columns=['model', 'accuracy', 'recall', 'precision', 'f1'])])
    print(f"\nFinished processing TF-IDF.\n")

    # printing results for each model
    print("\nGlobal results :")
    print(model_scores)

    # add column with subject count to model_scores
    model_scores['subject_limit'] = n_subjects
    # add column with number of news to model_scores
    model_scores['news_count'] = data_size  
    # add column with number of classes to predict to model_scores
    model_scores['predicition_class_count'] = len(news['subjects_encoded'].iloc[0])

    # SAVING RESULTS
    # check if the output file already exists
    if os.path.isfile(path_output_scores):
        # append the DataFrame without writing column names
        model_scores.to_csv(path_output_scores, mode="a", header=False, index=False)
    else:
        # save the DataFrame with column names
        model_scores.to_csv(path_output_scores, index=False)

    # add an empty row at the end of the CSV file (for readability)
    with open(path_output_scores, "a") as f:
        f.write("\n")

    print("\n\nFinished model and baseline evaluation !")
    print(f"Results saved in '{path_output_scores}'.")


if __name__ == '__main__':
    main()