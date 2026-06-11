import joblib

from preprocess import PreProcessor
from ner import Ner
from sentiment import Sentiment

class Test:
    def __init__(self):

        self.model = joblib.load(r"C:\Main_Drive\work\workarea\projects\news_article_analyzer\models\Baseline_logistic.pkl")

        self.vector = joblib.load(r"C:\Main_Drive\work\workarea\projects\news_article_analyzer\artifacts\tf_idf_vector.pkl")
        
        self.obj_pre = PreProcessor("","")

        self.obj_ner = Ner("")

    def run(self):

        article = input("Enter News Article : ")

        preprocessed_article = self.obj_pre.preprocess_text(article)

        tf_idf_article = self.vector.transform([preprocessed_article])

        model_predict = self.model.predict(tf_idf_article)

        print("Prediction :", model_predict)

        self.obj_ner = Ner(article)
        
        # self.obj_ner.ne_chunk_entities()

        self.obj_ner.spacy_ner()
        self.obj_ner.spacy_entities_freq()

        self.obj_senti = Sentiment(article)
        self.obj_senti.vader_sentiment()
        self.obj_senti.sentiment_label()



        


        

    
obj_test = Test()
obj_test.run()

