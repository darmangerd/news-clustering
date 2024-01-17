import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
from datetime import datetime
import time

# CONFIGURATION
# fix random seed for reproducibility
SEED: int = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# PARAMETERS
# paths
PATH_DATA_PAIRS: str = os.path.join("data", "pairs") # path to the data (pairs_{}.csv)
PATH_OUTPUT_SCORES: str = os.path.join("scores", "models_scores.csv") # path to save the scores
# parameters of the script
DATA_SIZE: int = 200 # size of the dataset to use
LABEL: str = "similarity_cosine" # choose label to use, either 'similarity_cosine', 'similarity_jaccard'
# hyperparameters to tune
EPOCHS: int = 8 # number of epochs to train the model 
SCHEDULER: str = 'warmupLinear' # scheduler to change learning rate, either 'warmupConstant', 'warmupLinear', 'warmupCosine'

# DATA
data = pd.read_csv(os.path.join(PATH_DATA_PAIRS, f"pairs_{DATA_SIZE}.csv"))
print(f"\nTraining all models on {DATA_SIZE} pairs of news articles.")

# convert datas to InputExample format for model training
examples = []
for index, row in data.iterrows():
    examples.append(InputExample(texts=[row["news1"], row["news2"]], label=row[LABEL]))

# split data into train, test, and validation
train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=SEED)
train_examples, val_examples = train_test_split(train_examples, test_size=0.2, random_state=SEED)

# MODELS
# define the list of models to test
model_names = [
    'sentence-transformers/all-mpnet-base-v2', # best quality
    'sentence-transformers/all-MiniLM-L6-v2', # best speed
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'sentence-transformers/all-distilroberta-v1',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'sentence-transformers/paraphrase-albert-small-v2',
    'sentence-transformers/paraphrase-MiniLM-L3-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v1',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
]

# callback use to save model score after each epoch
class EvaluationCallback:
    """
    Callback to evaluate the model on the validation set after each epoch
    """
    def __init__(self) -> None:
        """
        Initialize the list of scores
        """
        self.scores = []

    def __call__(self, score: float, epoch: int, steps: int) -> None:
        """
        Save the score after each epoch
        """
        self.scores.append(score)

# TRAINING
# loop over the models to train
for model_name in model_names:
    # max_seq_length : max number of words in a sentence
    word_embedding_model = models.Transformer(model_name, max_seq_length=256) 
     # pooling model for sentence embedding
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # dense model 
    dense_model = models.Dense(
        # dimension of the input vector
        in_features=pooling_model.get_sentence_embedding_dimension(), 
         # dimension of the output vector
        out_features=256,
        # activation function (tanh, relu, ...)
        activation_function=nn.Tanh()) 
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    # create dataloader to train (generator to load data in batches)
    train_dataloader = DataLoader(train_examples, batch_size=16)

    # define loss function : CosineSimilarityLoss (either 'CosineSimilarityLoss', 'MSELoss', 'ContrastiveLoss', 'SoftmaxLoss')
    train_loss = losses.CosineSimilarityLoss(model)

    # validation evaluator
    evaluator_val = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sts-val')
    # test evaluator
    evaluator_test = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')

    # initialize callback for saving model scores
    callback = EvaluationCallback()

    # define warmup steps based on data size, it's the number of steps before the learning rate starts to decrease (10% of train data)
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

    # start timer to measure training time
    start_time = time.time()

    print(f"\nStart training '{model_name}' model for {EPOCHS} epochs...")

    # train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        scheduler=SCHEDULER,
        evaluator=evaluator_val,
        callback=callback,
    )

    # end timer
    training_time = time.time() - start_time

    # evaluate the model
    evaluation_score = model.evaluate(evaluator_test)

    # get the best score
    best_score = model.best_score

    # create a dictionary with information about the training
    informations = {
        "date": datetime.now(),
        "model": model_name,
        "training_score": best_score,
        "evaluation_score": evaluation_score,
        "training_time": training_time,
        "scores_epochs" : [callback.scores],
        "data_size": DATA_SIZE,
        "label": LABEL,
        "epochs": EPOCHS,
        "warmup_steps": warmup_steps,
        "scheduler": SCHEDULER,
    }

    # convert the dictionary to a DataFrame
    informations_df = pd.DataFrame(informations, index=[0])

    # append the DataFrame to the output CSV file
    if os.path.isfile(PATH_OUTPUT_SCORES):
        informations_df.to_csv(PATH_OUTPUT_SCORES, mode="a", header=False, index=False)
    else:
        informations_df.to_csv(PATH_OUTPUT_SCORES, index=False)

    # print the best score
    print(f"Best score from training : {best_score}.")
    print(f"Model '{model_name}' tested.\n")

    # free memory
    del model
    del train_dataloader
    del evaluator_val
    del evaluator_test
    del callback
    del informations
    del informations_df

# put a empty row at the end of the CSV file
with open(PATH_OUTPUT_SCORES, "a") as f:
    f.write("\n")

print(f"\nAll models tested !\nScores saved in '{PATH_OUTPUT_SCORES}' file.")