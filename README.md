# Text-Classification-with-BERT

This is an implementation of text classification with BERT using PyTorch and the Hugging Face Transformers library. The code includes modules for data preprocessing, model training, and inference.

Installation

First, clone the repository:

$ git clone https://github.com/your_username/text-classification-with-bert.git
$ cd text-classification-with-bert

Then, install the required packages using pip:

$ pip install -r requirements.txt

Usage
1. Preprocessing
To preprocess the data, use the preprocessing.py module. The module reads in the data from a TSV file, preprocesses it, and returns a PyTorch Dataset. Here is an example:

from preprocessing import read_data, preprocess_data, TextClassificationDataset, get_data_loaders

# Read and preprocess the data
train_data = preprocess_data(read_data("train.tsv"))
dev_data = preprocess_data(read_data("dev.tsv"))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Create data loaders
train_loader, dev_loader = get_data_loaders(train_data, dev_data, tokenizer, batch_size=16)

2. Training
To train the model, use the train.py module. The module trains the model on the training set and evaluates it on the development set. Here is an example:

from model import TextClassifier
from train import train

# Define constants
TRAIN_FILE = "train.tsv"
DEV_FILE = "dev.tsv"
MODEL_NAME = "bert-base-cased"
NUM_LABELS = 3
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_PROPORTION = 0.1
WEIGHT_DECAY = 0.01

# Read and preprocess data
train_data = preprocess_data(read_data(TRAIN_FILE))
dev_data = preprocess_data(read_data(DEV_FILE))

# Load the tokenizer
tokenizer = TextClassifier(MODEL_NAME, NUM_LABELS, MAX_LEN).tokenizer

# Create data loaders
train_loader, dev_loader = get_data_loaders(train_data, dev_data, tokenizer, batch_size=BATCH_SIZE)

# Train the model
model = TextClassifier(MODEL_NAME, NUM_LABELS, MAX_LEN).model
model = train(model, train_loader, dev_loader, EPOCHS, BATCH_SIZE, LEARNING_RATE, WARMUP_PROPORTION, WEIGHT_DECAY)

# Save the model
torch.save(model.state_dict(), "text_classifier.pt")

3. Inference
To use the trained model for inference, use the predict.py module. The module takes in a text string and returns the predicted label. Here is an example:

from model import TextClassifier
from predict import predict

# Load the model
model = TextClassifier("bert-base-cased", 3, 256).model
model.load_state_dict(torch.load("text_classifier.pt"))

# Predict label for text
text = "This is an example sentence."
predicted_label = predict(model, text)

print(predicted_label)


Acknowledgements

"This project was inspired by the Hugging Face Transformers library and the official PyTorch tutorial on fine-tuning a BERT model for sequence classification."


4. Token Classification
To perform token classification with BERT, use the token_classification.py module. The module takes in a text string and a list of entity labels, and returns the text with the entities censored. Here is an example:

from token_classification import censor_entities

# Load the model
model_name = "dslim/bert-base-NER"
model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Censor entities in text
text = "John's phone number is 555-1234 and his email is john@example.com."
entity_labels = ["PER", "PHONE", "EMAIL"]
censored_text = censor_entities(model, tokenizer, text, entity_labels)

print(censored_text)

5. Running the Web Application
To run the web application, use the run.py module. The module starts a Flask web server that serves a web page for text classification. Here is an example:

from flask import Flask, render_template, request
from model import TextClassifier
import torch

app = Flask(__name__)

# Load the model
model = TextClassifier("bert-base-cased", 3, 256).model
model.load_state_dict(torch.load("text_classifier.pt"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        predicted_label = model.predict(text)
        return render_template("result.html", text=text, predicted_label=predicted_label)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    
    
References

Hugging Face Transformers
PyTorch Tutorial: Fine-tuning a BERT model for sequence classification






