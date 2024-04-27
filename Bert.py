from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast, EarlyStoppingCallback
from autocorrect import Speller
spell = Speller(lang='en')

class BertClassification:
    model_path = "model"

    def __init__(self):
        self.model=BertForSequenceClassification.from_pretrained(self.model_path,use_auth_token=True)
        self.tokenizer=BertTokenizerFast.from_pretrained(self.model_path,use_auth_token=True)
        self.pipeline=pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def get_prediction(self, text):
        text = spell(text)
        result=self.pipeline(text)
        return result



