import pandas as pd
import numpy as np
import nltk
import re
import string
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download("punkt")  
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")



class PreProcessor:
    def __init__(self, train_ds_path, test_ds_path):
        self.train_path = train_ds_path
        self.test_path = test_ds_path
        self.stop_words = set(stopwords.words("english"))
        self.lemma = WordNetLemmatizer()



    def load_ds(self):
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        self.train_df.columns = ["label", "title", "description"]
        self.test_df.columns = ["label", "title", "description"]

    def preprocess_text(self, text):

        # lower case
        text = text.lower()

        # remove characters
        # text = text.translate((str.maketrans("","",string.punctuation)))
        text = "".join([char for char in text if char not in string.punctuation])

        # remove digits
        # text = re.sub(r"\d+","",text)
        text = "".join([char for char in text if char not in string.digits])

        # Tokens
        words = word_tokenize(text)

        # stopwords remove
        words = [w for w in words if w not in self.stop_words]

        # converting words into their base form (lemmatize)
        words = [self.lemma.lemmatize(w) for w in words]

        return " ".join(words)

    def preprocess_dataset(self):

        self.train_df["text"] = (self.train_df["title"] + " " + self.train_df["description"])
        self.test_df["text"] = self.test_df["title"] + " " + self.test_df["description"]


        self.train_df["clean_text"] = self.train_df["text"].apply(self.preprocess_text)
        self.test_df["clean_text"] = self.test_df["text"].apply(self.preprocess_text)

    

    def save_processed_ds(self):

        self.train_df.to_csv(r"data\processed\train_processed.csv", index=False)
        self.test_df.to_csv(r"data\processed\test_processed.csv", index=False)

        print("Done")



