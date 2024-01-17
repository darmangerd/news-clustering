import pandas as pd
import numpy as np
import os
import gzip
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import csv
from difflib import SequenceMatcher
import os.path


class NewsProcessor:
    """
    The NewsProcessor class is designed to process and analyze news data. It provides methods to ensure the 
    correctness and formatting of news data for analysis purposes, including removing inappropriate news, handling 
    the number of subjects, eliminating duplicates, and more. Additionally, it offers functionality to 
    generate data pairs and calculate Jaccard and cosine similarities between news pairs.

    Attributes:
        lang (list): The list of languages to filter the news data.
        columns (list): The list of columns to keep in the news data.
        news (list): The list of news data.
        df (DataFrame): The DataFrame of news data.
        df_codes (DataFrame): The DataFrame of news codes.
        valid_subjects (set): The set of valid subjects.
        df_encoded (DataFrame): The DataFrame of news data with encoded subjects, having binary columns for each subject indicating whether a news has that subject or not. 
    """
    
    def __init__(self, lang: list = ['en', 'EN'], columns: list = ['guid', 'data.versionCreated', 'data.headline', 'data.body', 'data.subjects']) -> None:
        """
        Constructor method.

        Args:
            lang (list): The list of languages to filter the news data.
            columns (list): The list of columns to keep in the news dataframe.
        """
        self.lang = lang
        self.columns = columns
        self.news = None
        self.df = None
        self.df_codes = None
        self.valid_subjects = None
        self.df_encoded = None


    def load_news_codes(self, path: str) -> None:
        """
        Load the news codes file into a DataFrame and sets the index for quick access. 
        The codes are used later for filtering valid subjects.

        Args:
            path (str): The path to the news codes file.
        """
        self.df_codes = pd.read_csv(path, sep=';')
        self.df_codes.set_index('Code', inplace=True)
        self.valid_subjects = set(self.df_codes.index)
        

    def load_news(self, directory: str) -> None:
        """
        Load news data from JSON files in a given directory. It iterates over the files, reads 
        each file using pd.read_json, and concatenates the resulting DataFrame to the main DataFrame (self.df).

        Args:
            directory (str): The directory containing the json files.
        """
        # initialize the dataframe (empty)
        self.df = pd.DataFrame()

        # iterate over all json files in the directory
        for filename in os.listdir(directory):
            # check the correct file extension
            if filename.endswith(".txt.gz"):
                filepath = os.path.join(directory, filename)
                # read the json file
                with gzip.open(filepath, 'r') as f:
                    news = pd.read_json(f)

                # append the data from the current file to main dataframe (self.df)
                self.df = pd.concat([self.df, pd.json_normalize(news['Items'])])

        
    def filter_lang(self, lang: list = ['en', 'EN']) -> None:
        """
        Filter the news data based on the specified languages.
        """
        self.df = self.df[self.df['data.language'].isin(self.lang)]
        

    def keep_columns(self) -> None:
        """
        Select and rename specific columns from the DataFrame and drop the rest. 
        Use the columns from the 'columns' attribute.
        """
        self.df = self.df[self.columns]
        self.df.rename(columns={'data.versionCreated': 'date', 'data.headline': 'headline', 'data.body': 'body', 'data.subjects': 'subjects'}, inplace=True)
        

    def drop_duplicates(self) -> None:
        """
        Drop duplicate rows in the data based on the 'headline' and 'guid' columns.
        """
        self.df.drop_duplicates(subset=['headline'], inplace=True)
        self.df.drop_duplicates(subset=['guid'], inplace=True)

        
    def filter_valid_subjects(self) -> None:
        """
        Filter the 'subjects' column for each news to include only valid subjects based on the news codes file. 
        It checks each subject in the 'subjects' list and keeps only those that exist in the set of valid subjects.
        """
        # check that each subject is in the set of valid subjects
        self.df['subjects'] = self.df['subjects'].apply(lambda subjects: list(set([s for s in subjects if s in self.valid_subjects])))


    def filter_unique_subjects_per_news(self) -> None:
        """
        Filter the 'subjects' column for each news to include only unique subjects. Also, convert the 'subjects' column to a list.
        """
        # use set function to remove duplicates
        self.df['subjects'] = self.df['subjects'].apply(lambda subjects: list(set(subjects)))


    def sort_subjects(self) -> None:
        """ 
        Sort the 'subjects' for each news alphabetically. This ensures that the same subjects are represented in the same order, making it 
        easier to compare subjects across news.
        """
        self.df['subjects'] = self.df['subjects'].apply(lambda subjects: sorted(subjects))


    def filter_subjects_count(self) -> None:
        """
        Count the number of subjects for each news and keep only the news with subjects count within the interquartile range (IQR).
        """
        # create a momentary column with the number of subjects per news
        self.df['subjects_length'] = self.df['subjects'].apply(lambda subjects: len(subjects))

        # calculate the interquartile range (IQR)
        Q1 = self.df['subjects_length'].quantile(0.25)
        Q3 = self.df['subjects_length'].quantile(0.75)

        # calculate the IQR
        IQR = Q3 - Q1

        # filter out the news with subjects length outside the IQR
        self.df = self.df[(self.df['subjects_length'] >= Q1 - 1.5 * IQR) & (self.df['subjects_length'] <= Q3 + 1.5 * IQR)]

        # drop the 'subjects_length' column
        self.df.drop(columns=['subjects_length'], inplace=True)


    def filter_subject_appearances(self, min_appearances: int, max_appearances: int) -> None:
        """
        Filter out the subjects that appear less than min_appearances or more than max_appearances.

        Args:
            min_appearances (int): The minimum number of times a subject must appear to be kept.
            max_appearances (int): The maximum number of times a subject must appear to be kept.

        References:
            https://docs.python.org/3/library/collections.html
        """
        # flatten the subjects list to get a list of all subjects in the dataframe
        subjects = [s for sublist in self.df['subjects'].tolist() for s in sublist]
        # count the number of times each subject appears
        subject_counts = Counter(subjects)

        # find subjects that are to be filtered out (appear less than min_appearances or more than max_appearances)
        subjects_to_filter = {s for s, count in subject_counts.items() 
                            if count < min_appearances or count > max_appearances}
        
        # filter out the subjects
        self.df['subjects'] = self.df['subjects'].apply(lambda subjects: [s for s in subjects if s not in subjects_to_filter])

            
    def filter_empty_subjects(self) -> None:
        """
        Filter out the news with empty 'subjects'.
        """
        # keep only the news with subjects list length greater than 0
        self.df = self.df[self.df['subjects'].map(len) > 0]


    def preprocess(self, file: str, min_appearances: int, max_appearances: int) -> None:
        """
        Preprocess the news data by applying all the preprocessing steps.

        In details, it combines all the preprocessing steps in a specific order. It calls the relevant methods to load 
        news data, filter by language, keep columns, drop duplicates, filter valid subjects, filter unique subjects, 
        sort subjects, filter subjects count, filter subject appearances, filter empty subjects, and reset the DataFrame index.

        Args:
            file (str): The path to the news data file.
            min_appearances (int): The minimum number of times a subject must appear to be kept.
            max_appearances (int): The maximum number of times a subject must appear to be kept.
        """
        print("Preprocessing news data...")

        # load the news data
        self.load_news(file)
        print(f"Size before filtering: {self.df.shape}.")

        # apply the preprocessing steps
        self.filter_lang()
        self.keep_columns()
        self.drop_duplicates()
        self.filter_valid_subjects()
        self.filter_unique_subjects_per_news()
        self.sort_subjects()
        self.filter_subject_appearances(min_appearances, max_appearances)
        self.filter_subjects_count()
        self.filter_empty_subjects()

        # reset the index
        self.df.reset_index(drop=True, inplace=True)

        print(f"Size after filtering: {self.df.shape}.")
        print("Preprocessing complete.")


    def preprocess_debug(self, file: str, min_appearances: int, max_appearances: int) -> None:
        """
        Method is similar to `preprocess`, but it prints the size of the DataFrame after 
        each preprocessing step. It provides additional information for debugging and testing purposes.

        In details, it combines all the preprocessing steps in a specific order. It calls the relevant methods to load 
        news data, filter by language, keep columns, drop duplicates, filter valid subjects, filter unique subjects, 
        sort subjects, filter subjects count, filter subject appearances, filter empty subjects, and reset the DataFrame index.

        Args:
            file (str): The path to the news data file.
            min_appearances (int): The minimum number of times a subject must appear to be kept.
            max_appearances (int): The maximum number of times a subject must appear to be kept.
        """
        # load the news data
        print("Preprocessing news data... (debug mode)")
        self.load_news(file)

        # apply the preprocessing steps
        print(f"Size before filtering: {self.df.shape}.")
        self.filter_lang()
        print(f"Size after filtering language: {self.df.shape}.")
        self.keep_columns()
        print(f"Size after keeping columns: {self.df.shape}.")
        self.drop_duplicates()
        print(f"Size after dropping duplicates: {self.df.shape}.")
        self.filter_valid_subjects()
        print(f"Size after filtering valid subjects: {self.df.shape}.")
        self.filter_unique_subjects_per_news()
        print(f"Size after filtering unique subjects per news: {self.df.shape}.")
        self.sort_subjects()
        print(f"Size after sorting subjects: {self.df.shape}.")
        self.filter_subject_appearances(min_appearances, max_appearances)
        print(f"Size after filtering subjects appearances: {self.df.shape}.")
        self.filter_subjects_count()
        print(f"Size after filtering subjects count: {self.df.shape}.")

        print("Subjects value counts before filtering empty subjects: ")
        # print value counts of subjects
        self.df.reset_index(drop=True, inplace=True)
        # flatten the subjects list to get a list of all subjects in the dataframe
        subjects = [s for sublist in self.df['subjects'].tolist() for s in sublist]
        # count the number of times each subject appears
        subject_counts = pd.Series(subjects).value_counts()
        print("Head of subjects value counts: ")
        print(subject_counts.head(5))
        print("Tail of subjects value counts: ")
        print(subject_counts.tail(5))
    
        self.filter_empty_subjects()
        print(f"Size after filtering empty subjects: {self.df.shape}.")

        print("Subjects value counts after filtering empty subjects: ")
        # print value counts of subjects
        self.df.reset_index(drop=True, inplace=True)
        # flatten the subjects list to get a list of all subjects
        subjects = [s for sublist in self.df['subjects'].tolist() for s in sublist]
        # count the number of times each subject appears
        subject_counts = pd.Series(subjects).value_counts()
        print("Head of subjects value counts: ")
        print(subject_counts.head(5))
        print("Tail of subjects value counts: ")
        print(subject_counts.tail(5))

        print("Preprocessing complete. (debug mode)")

        
    def process(self, file_code: str, file_news: str, min_appearances: int = 100, max_appearances: int = np.inf, debug: bool = False) -> None:
        """
        Complete the processing of the news data including loading the news codes, and preprocessing the news data. 
        The default values for subjects apppearances are set based on the distribution of 
        the training data (for more information, see the notebook `data_exploration.ipynb`).  

        Args:
            file_code (str): The path to the news codes file.
            file_news (str): The path to the news data file.
            min_appearances (int): The minimum number of times a subject must appear to be kept.
            max_appearances (int): The maximum number of times a subject must appear to be kept.
            debug (bool): If True, print the size of the data after each step.
        """
        # load the news codes
        self.load_news_codes(file_code)
        # preprocess the news data
        if debug:
            self.preprocess_debug(file_news, min_appearances, max_appearances)
        else:
            self.preprocess(file_news, min_appearances, max_appearances)       


    def get_news(self) -> pd.DataFrame:
        """
        Return the news data.

        Returns:
            DataFrame: The news data.
        """
        return self.df
    
    
    def get_news_codes(self) -> pd.DataFrame:
        """
        Return the news codes.

        Returns:
            DataFrame: The news codes.
        """
        return self.df_codes
    
    
    def create_random_pairs_headline(self, num_pairs: int) -> pd.DataFrame:
        """
        Generates random pairs of news headlines from the DataFrame. It randomly selects two different headlines and 
        ensures that they are not repeated as (a, b) and (b, a) pairs. The method returns a DataFrame of pairs.

        Args:
            num_pairs (int): The number of pairs to generate.

        Returns:
            DataFrame: A DataFrame of pairs of news headlines.
        """
        # get all the headlines
        headlines = self.df['headline'].values

        # make sure we don't try to take more pairs than possible
        max_pairs = len(headlines) * (len(headlines) - 1) // 2 # we divide by 2 because we don't want to count (a, b) and (b, a) as different pairs
        num_pairs = min(num_pairs, max_pairs) # if num_pairs > max_pairs, we take max_pairs

        # create a set of pairs to make sure we don't take the same pair twice
        pairs_set = set()
        while len(pairs_set) < num_pairs:
            # sorted and replace=False to make sure we don't take (a, b) and (b, a) as different pairs
            pair = tuple(sorted(np.random.choice(headlines, size=2, replace=False))) 
            pairs_set.add(pair)
        
        pair_df = pd.DataFrame(list(pairs_set), columns=['news1', 'news2'])

        return pair_df
    

    def select_top_subjects_per_news(self, n: int = 5) -> None:
        """
        Select the top `n` subjects per news based on their frequency in the dataset. It uses the `Counter` class to count the frequency 
        of each subject and selects the top n subjects per news.

        Args:
            n (int): The number of subjects to keep per news.
        """
        # flatten the subjects list for each news
        all_subjects = [s for sublist in self.df['subjects'].tolist() for s in sublist]

        # count the frequency of each subject in the dataset
        subject_counts = Counter(all_subjects)

        # function to get the frequency of a subject
        def get_subject_frequency(subject):
            return subject_counts[subject]

        # select the top n subjects per news based on their frequency in the dataset
        self.df['subjects'] = self.df['subjects'].apply(lambda subjects: sorted(subjects, key=get_subject_frequency, reverse=True)[:n])


    def encode_subjects(self) -> pd.DataFrame:
        """
        Encode the 'subjects' column using MultiLabelBinarizer. 
        It creates binary columns for each subject, indicating whether a news has that subject or not.

        Returns:
            df_encoded (DataFrame): The dataframe with the encoded subjects, having binary columns for each subject indicating whether a news has that subject or not.
        """
        # get all subjects from the dataframe and sort them to make sure the order is always the same 
        all_subjects = sorted([s for sublist in self.df['subjects'].tolist() for s in sublist])
        mlb = MultiLabelBinarizer()
        # fit the MultiLabelBinarizer on all subjects
        mlb.fit([all_subjects])
        # encode the subjects of each news
        self.df_encoded = mlb.transform(self.df['subjects'].tolist())
        return self.df_encoded
    

    def jaccard_similarity(self, subjects1: list, subjects2: list) -> float:
        """
        Calculate  calculates the Jaccard similarity between two sets of subjects. It uses the jaccard_score function 
        from scikit-learn with the 'macro' average. This function is used to calculate the Jaccard similarity between 
        labels and predictions. In our case, we use the subjects from the first news as the labels and the subjects from
        the second news as the predictions. The 'macro' average calculates the Jaccard similarity for each subject separately
        and then takes the unweighted mean of the Jaccard similarities. 

        Example:
            subjects1 = ['a', 'b', 'c'], subjects2 = ['a', 'b', 'd']
            jaccard_score(subjects1, subjects2, average='macro') = (1 + 1 + 0) / 3 = 0.66

            - the Jaccard similarity for 'a' is 1 because 'a' is in both subjects1 and subjects2
            - the Jaccard similarity for 'b' is 1 because 'b' is in both subjects1 and subjects2
            - the Jaccard similarity for 'c' is 0 because 'c' is in subjects1 but not in subjects2
            - the Jaccard similarity for 'd' is 0 because 'd' is in subjects2 but not in subjects1
            
            The 'macro' average is the unweighted mean of the Jaccard similarities for each subject in this case, the 'macro' 
            average is (1 + 1 + 0 + 0) / 4 = 0.5
                
            
        Args:
            subjects1 (list): A list of subjects from the first news.
            subjects2 (list): A list of subjects from the second news.

        Returns:
            float: The Jaccard similarity between the two subject sets.

        Reference:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
        """
        return jaccard_score(subjects1, subjects2, average='macro') 
    

    def cosine_similarity(self, subjects1: list, subjects2: list) -> float:
        """
        Calculate the cosine similarity between two sets of subjects. It uses the cosine_similarity function from 
        scikit-learn to compute the similarity between two vectors.

        Args:
            subjects1 (list): A list of subjects from the first news.
            subjects2 (list): A list of subjects from the second news.

        Returns:
            float: The cosine similarity between the two subject sets.

        Reference:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        """
        # cosine_similarity returns a 2D array (kernel matrix), so we take the first element of the first row to get the similarity value 
        return cosine_similarity([subjects1], [subjects2])[0][0] 


    def calculate_similarities(self, pairs: pd.DataFrame, use_jaccard: bool = True, use_cosine: bool = True) -> pd.DataFrame:
        """
        Calculates the Jaccard and/or cosine similarity for each pair of news headlines. Adds new columns for 
        the similarities to the pairs DataFrame.

        Args:
            pairs (DataFrame): A DataFrame containing pairs of news headlines.
            use_jaccard (bool): Indicates whether to calculate Jaccard similarity.
            use_cosine (bool): Indicates whether to calculate cosine similarity.

        Returns:
            DataFrame: A new DataFrame with new columns for the Jaccard and/or cosine similarity of each pair.
        """
        if use_jaccard:
            # calculates the Jaccard similarity for each pair of news headlines by applying the jaccard_similarity function to each row of the pairs DataFrame
            pairs['similarity_jaccard'] = pairs.apply(lambda row: self.jaccard_similarity(
                # creates a boolean by comparing each 'headline' in self.df with the 'news1' value in the current row. Uses this boolean to filter self.df_encoded
                self.df_encoded[self.df['headline'] == row['news1']].tolist()[0],  # converts the filtered DataFrame into a list and takes the first element (the list contains only one element)
                self.df_encoded[self.df['headline'] == row['news2']].tolist()[0]),  # does the same for 'news2'
                                                    axis=1)
        if use_cosine:
            # calculates the cosine similarity for each pair of news headlines using the same approach as above (for Jaccard similarity)
            pairs['similarity_cosine'] = pairs.apply(lambda row: self.cosine_similarity(
                self.df_encoded[self.df['headline'] == row['news1']].tolist()[0],
                self.df_encoded[self.df['headline'] == row['news2']].tolist()[0]),
                                                    axis=1)
        return pairs


    def filter_similar_headlines(self, pairs: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
        """
        Filter out pairs of news that have similar headlines based on a similarity threshold. It calculates the similarity between the 
        headlines using the SequenceMatcher class from difflib and filters out pairs with similarity above the threshold.

        The ratio value returned by the SequenceMatcher class is a measure of the sequences’ similarity as a float in the range [0, 1]. The
        higher the ratio, the more similar the sequences are. The ratio is calculated as follows:
            ratio = 2.0 * M / T
            where M is the number of matches and T is the total number of elements in both sequences.

        Reference:
            https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
        
        Args:
            pairs (DataFrame): A DataFrame containing pairs of news headlines.
            similarity_threshold (float): The similarity threshold for headline similarity (between 0.0 and 1.0).

        Returns:
            DataFrame: The filtered pairs DataFrame.
        """
        filtered_pairs = pairs.copy()
        # calculate the similarity between the headlines of each pair of news by applying the SequenceMatcher class to each row of the pairs dataframe
        filtered_pairs['headline_similarity'] = filtered_pairs.apply(
            lambda row: SequenceMatcher(None, row['news1'], row['news2']).ratio(), axis=1)
        # filter out pairs with similarity above the threshold (that are too similar)
        filtered_pairs = filtered_pairs[filtered_pairs['headline_similarity'] < similarity_threshold]
        filtered_pairs.drop(columns=['headline_similarity'], inplace=True)
        return filtered_pairs
    

    def generate_pairs(self, num_pairs: int, filepath: str, use_jaccard: bool = True, use_cosine: bool = True,
                       headline_similarity_threshold: float = 0.90) -> None:
        """
        Create random pairs of news headlines, calculates the similarities, filters out similar headlines, and saves the results to a CSV file. 
        It provides flexibility to enable/disable Jaccard similarity, cosine similarity, and headline similarity filtering.

        Args:
            num_pairs (int): The number of pairs to create. The actual number of pairs created may be lower if there are not enough unique news headlines.
            filepath (str): The filepath to save the results to.
            use_jaccard (bool): Whether to calculate Jaccard similarity or not.
            use_cosine (bool): Whether to calculate cosine similarity or not.
            headline_similarity_threshold (float): The similarity threshold for headline similarity (between 0.0 and 1.0).
        """
        print(f"\n\nGenerating {num_pairs} pairs of news headlines...")

        # encode the subjects of the news headlines
        self.encode_subjects()
        # create random pairs of news headlines
        pairs = self.create_random_pairs_headline(num_pairs)
        # calculate the similarities between the pairs of news headlines
        pairs = self.calculate_similarities(pairs, use_jaccard, use_cosine)
        # filter out pairs of news that have similar headlines
        pairs = self.filter_similar_headlines(pairs, similarity_threshold=headline_similarity_threshold)

        # add columns 'guid' to the pairs dataframe
        pairs['guid1'] = pairs.apply(lambda row: self.df[self.df['headline'] == row['news1']]['guid'].values[0], axis=1)
        pairs['guid2'] = pairs.apply(lambda row: self.df[self.df['headline'] == row['news2']]['guid'].values[0], axis=1)

        # reorder the columns for better readability (guids first, then news, then similarities)
        if use_jaccard and use_cosine:
            pairs = pairs[['guid1', 'news1', 'guid2', 'news2', 'similarity_jaccard', 'similarity_cosine']]
        elif use_jaccard:
            pairs = pairs[['guid1', 'news1', 'guid2', 'news2', 'similarity_jaccard']]
        elif use_cosine:
            pairs = pairs[['guid1', 'news1', 'guid2', 'news2', 'similarity_cosine']]

        print(f"\nNumber of pairs generated after filtering similar headlines: {len(pairs)}")

        # rename the columns (for better readability)
        pairs = pairs.rename(columns={'guid1': 'guid_news1', 'guid2': 'guid_news2'})

        # save the pairs to a CSV file
        # quoting=csv.QUOTE_NONNUMERIC is used to avoid issues with commas in the news headlines
        pairs.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)  
        print(f"\nSaved results to '{filepath}' file.")