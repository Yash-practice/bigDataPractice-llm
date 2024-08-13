from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from scipy.special import softmax

def load_roberta_model(model, tokenizer):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
    model = RobertaForSequenceClassification.from_pretrained(model)
    return tokenizer,model

def predict_sentiment(text, model, tokenizer, sentiment_mapping):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output = model(**inputs)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return {
        'output' : sentiment_mapping[ranking[0]],
        'probs' : {sentiment_mapping[ranks]: scores[ranks] for ranks in ranking}
    }