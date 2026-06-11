from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Sentiment:
    def __init__(self,text):
        self.text = text
        self.analyzer = SentimentIntensityAnalyzer()

    def vader_sentiment(self):
        self.sentiment_score = self.analyzer.polarity_scores(self.text)
        
        self.compound_score = self.sentiment_score["compound"]

        print(self.sentiment_score)
        return self.sentiment_score
    
    def sentiment_label(self):
        if self.compound_score >= 0.05:
            print("Positive")

        elif self.compound_score <= -0.05:
            print("Negative")
        
        else:
            print("Neutral")
            
