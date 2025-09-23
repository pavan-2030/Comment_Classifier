from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert_sentiment_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_sentiment_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    logits = model(inputs).logits
    probs = tf.nn.softmax(logits, axis=-1)
    label_id = tf.argmax(probs, axis=1).numpy()[0]
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[label_id], probs.numpy()[0]

#text = input("Enter a comment: ")
#print(predict(text))