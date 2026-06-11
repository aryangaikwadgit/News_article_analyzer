import pandas as pd

class Utils:

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def save_data(self, obj, file_path):
        pass

    def get_lowercase(self, text):
        return text.lower()