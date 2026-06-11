from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud


import joblib



class FeatureExtractor:
    def __init__(self,train_df,test_df):
        self.train_df = train_df
        self.test_df = test_df 

    def word_count(self):
        all_words = " ".join(self.train_df["clean_text"])
        all_words = all_words.split()
        word_count = Counter(all_words)
        print(word_count.most_common(5))

    def word_cloud(self):

        text = " ".join(self.train_df["clean_text"])

        wc = WordCloud(width= 800,height=400,background_color="white").generate(text)

        wc.to_file(r"artifacts\word_cloud.png")
        print("Word Cloud Saved")

    def bag_of_words(self):
        self.cv = CountVectorizer(max_features=5000)

        self.x_train = self.cv.fit_transform(self.train_df["clean_text"])
        self.x_test = self.cv.transform(self.test_df["clean_text"])

        self.y_train = self.train_df["label"]
        self.y_test = self.test_df["label"]
      
        # print("BoW Sample Labels")
        # print("y_train :", self.x_train.iloc[10])
        # print("y_test :", self.x_test.iloc[10])
        
        print("BoW Shape")
        print(self.x_train.shape)
        print(self.x_test.shape)

        return self.x_train,self.x_test,self.y_train,self.y_test 
        
    def tf_idf(self):
        self.tv = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

        self.x_train = self.tv.fit_transform(self.train_df["clean_text"])
        self.x_test = self.tv.transform(self.test_df["clean_text"])

        self.y_train = self.train_df["label"]
        self.y_test = self.test_df["label"]

        # print("tf_idf Sample Labels")
        # print("y_train :", self.x_train.iloc[10])
        # print("y_test :", self.x_test.iloc[10])
        
        print("tf_idf Shape")
        print(self.x_train.shape)
        print(self.x_test.shape)


        joblib.dump(self.tv,r"artifacts\tf_idf_vector.pkl")        
        print("Ready For Model Training")

        return self.x_train, self.x_test, self.y_train, self.y_test
    