from src.preprocess import PreProcessor
from src.feature_extraction import FeatureExtractor
from src.baseline_model import Model



class NewsArticleAnalzer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.obj_pre = PreProcessor(self.train_path, self.test_path)
        self.obj_fe = None
        self.obj_model = None


    def run(self):
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



if __name__ == "__main__":
    train_path = r"data\raw\train.csv"
    test_path = r"data\raw\test.csv"
    obj_news = NewsArticleAnalzer(train_path, test_path)
    obj_news.run()
