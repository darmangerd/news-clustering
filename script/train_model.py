# REFERENCE : https://www.sbert.net/docs/training/overview.html, https://www.sbert.net/docs/pretrained_models.html

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# CONFIGURATION
# fix random seed for reproducibility for comparison
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# PARAMETERS
# paths
PATH_DATA_PAIRS: str = os.path.join('data', 'pairs') # path to the pairs data for training (pairs_{}.csv)
PATH_MODEL_SAVE: str = os.path.join('model') # output path to save the model
# parameters of the script
DATA_SIZE: int = 200 # size of the dataset to use for training
MODEL_NAME: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' # pre-trained Sentence-Transformer model to use 
LABEL: str = "similarity_cosine" # choose label to use, either 'similarity_cosine', 'similarity_jaccard'
# hyperparameters to tune
EPOCHS: int = 8 # number of epochs to train the model 
SCHEDULER: str = 'warmupLinear' # scheduler to change learning rate, either 'warmupConstant', 'warmupLinear', 'warmupCosine'


def main(
    path_data_pairs: str = PATH_DATA_PAIRS,
    path_model_save: str = PATH_MODEL_SAVE,
    data_size: int = DATA_SIZE,
    model_name: str = MODEL_NAME,
    label: str = LABEL,
    epochs: int = EPOCHS,
    scheduler: str = SCHEDULER,
) -> None:
    """
    Train a model to compute similarity between news articles using a pre-trained model from Sentence-Transformers, by fine-tuning it on a dataset
    of pairs of news articles. The model is saved in the `path_model_save` folder. 

    Args:
        path_data_pairs (str): path to the data (pairs_{}.csv)
        path_model_save (str): path to save the model
        data_size (int): size of the dataset to use
        model_name (str): pre-trained model to use
        label (str): choose label to use, either 'similarity_cosine', 'similarity_jaccard'
        epochs (int): number of epochs to train the model
        scheduler (str): scheduler to change learning rate, either 'warmupConstant', 'warmupLinear', 'warmupCosine'
    """
    
    # MODEL
    # load pre-trained model from Sentence-Transformers
    # max_seq_length : max number of words in a sentence
    word_embedding_model = models.Transformer(model_name, max_seq_length=256) 
    # pooling layer for sentence embedding 
    pooling_layer = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # dense layer to compute similarity between news articles
    dense_layer = models.Dense(
        # dimension of the input vector
        in_features=pooling_layer.get_sentence_embedding_dimension(), 
         # dimension of the output vector
        out_features=256,
        # activation function (tanh, relu, ...)
        activation_function=nn.Tanh()) 
    # final model to train
    model = SentenceTransformer(modules=[word_embedding_model, pooling_layer, dense_layer]) 

    # DATA
    # load data from pairs_{}.csv
    data = pd.read_csv(os.path.join(path_data_pairs, f'pairs_{data_size}.csv'))
    print(f"\nNumber of pairs of news for training: {len(data)}.")

    # convert datas to InputExample format for model training
    examples = []
    for index, row in data.iterrows():
        examples.append(InputExample(texts=[row["news1"], row["news2"]], label=row[label]))

    # split data into train, test and validation
    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=SEED)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.2, random_state=SEED)

    # create dataloader to train (generator to load data in batches)
    train_dataloader = DataLoader(train_examples, batch_size=16)

    # LOSS FUNCTION AND EVALUATION
    # define loss function : CosineSimilarityLoss (either 'CosineSimilarityLoss', 'MSELoss', 'ContrastiveLoss', 'SoftmaxLoss')
    train_loss = losses.CosineSimilarityLoss(model)

    # train evaluator
    evaluator_train = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(train_examples, name='sts-train')
    # validation evaluator
    evaluator_val = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sts-val')  
    # test evaluator
    evaluator_test = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')  

    # MODEL TRAINING
    class EvaluationCallback:
        """
        Callback to evaluate the model on test, train and validation set at each epoch.
        """
        def __init__(self) -> None:
            """
            Initialize the callback to save the scores from test, train and validation set.
            """
            self.scores_val = []
            self.scores_test = []
            self.scores_train = []

        def __call__(self, score: float, epoch: int, steps: int) -> None:
            """
            Evaluate the model on test, train and validation set at each epoch.
            
            Args:
                score (float): score of the model on validation set
                epoch (int): current epoch
                steps (int): current step
            """
            # evaluate the model on validation set at each epoch
            self.scores_val.append(score)
            # evaluate the model on train set at each epoch
            score_train = model.evaluate(evaluator_train)
            self.scores_train.append(score_train)
            # evaluate the model on test set at each epoch
            score_test = model.evaluate(evaluator_test)
            self.scores_test.append(score_test)

    # initialize the callback to save the scores from test, train and validation set
    callback = EvaluationCallback()

    # define warmup steps based on data size, it's the number of steps before the learning rate starts to decrease (10% of train data)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    # train the model
    print(f"\nTraining the model '{model_name}' for {epochs} epochs ...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        evaluator=evaluator_val,
        output_path=path_model_save,
        save_best_model=True,
        callback=callback,    
    )
    
    # save training score evolution
    plt.plot(callback.scores_train, label='Train')
    plt.plot(callback.scores_test, label='Test')
    plt.plot(callback.scores_val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Score evolution from train, test and validation set')
    plt.legend()
    plt.savefig(os.path.join(path_model_save, 'scores_evaluation.png'))

    # evaluate on test set
    print(f"\nEvaluation score on test set: {str(model.evaluate(evaluator_test))}.")


if __name__ == '__main__':
    main()