import joblib

from src.preprocess import PreProcessor
from src.feature_extraction import FeatureExtractor
from src.baseline_model import Model
from src.ner import Ner
from src.sentiment import Sentiment


class NewsArticleAnalzer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.obj_pre = PreProcessor(self.train_path, self.test_path)
        self.obj_fe = None
        self.obj_model = None

    # ---------------- TRAIN ---------------- #

    def train(self):
        self.obj_pre.load_ds()
        self.obj_pre.preprocess_dataset()
        self.obj_pre.save_processed_ds()

        self.obj_fe = FeatureExtractor(self.obj_pre.train_df, self.obj_pre.test_df)

        self.obj_fe.word_count()
        self.obj_fe.word_cloud()
        self.obj_fe.bag_of_words()
        self.obj_fe.tf_idf()

        self.obj_model = Model(
            self.obj_fe.x_train,
            self.obj_fe.x_test,
            self.obj_fe.y_train,
            self.obj_fe.y_test,
        )

        self.obj_model.train_model()
        self.obj_model.evaluate_model()
        self.obj_model.save_model()

    # ---------------- TEST ---------------- #

    def analyze_article(self):

        model = joblib.load("models/Baseline_logistic.pkl")
        vector = joblib.load("artifacts/tf_idf_vector.pkl")

        article = input("Enter News Article : ")

        clean_article = self.obj_pre.preprocess_text(article)

        article_vector = vector.transform([clean_article])

        prediction = model.predict(article_vector)

        print("Prediction :", prediction)

        ner = Ner(article)

        ner.spacy_ner()
        ner.spacy_entities_freq()

        sentiment = Sentiment(article)

        sentiment.vader_sentiment()
        sentiment.sentiment_label()


if __name__ == "__main__":

    train_path = r"data\raw\train.csv"
    test_path = r"data\raw\test.csv"

    obj_news = NewsArticleAnalzer(train_path, test_path)

    # Train Model
    # obj_news.train()

    # Test New Article
    obj_news.analyze_article()
