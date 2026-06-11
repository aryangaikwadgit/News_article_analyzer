from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class Model:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        self.model = LogisticRegression()

        self.model.fit(self.x_train, self.y_train)

        joblib.dump(self.model, r"artifacts\model.pkl")

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.x_test)

        self.accuracy = accuracy_score(self.y_test, self.y_pred)

        class_report = classification_report(self.y_test, self.y_pred)
        print(class_report)

        with open(r"artifacts\classification_report.txt", "w") as file:
            file.write(class_report)
        print("Classification report saved")

        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        print(conf_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.savefig(r"artifacts\confusion_Matrix.png")
        plt.close()
        print("confusion matrix saved")

    def save_model(self):
        joblib.dump(self.model, r"models\Baseline_logistic.pkl")
