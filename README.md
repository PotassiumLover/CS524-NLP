# CS524 - NLP Project 2

#### Team: 7 

#### Author: G. K. Chesterton 

#### Task: Authorship Attribution

#### DUE: December 4

#### Installation

1. Requires a modern Python environment (3.12 is what was used but it should work with similar versions)
2. Requires Jupyter Notebook environment
3. Run build.sh to install all python requirements in requirements.txt and install spacy en_core_web_sm 

If the build script does not work, install the pip packages manually with `pip install -r requirements.txt` and install spacy package with `python3 -m spacy download en_core_web_sm`

#### Running the project

A total of five notebooks were created, each representing a different model used to train the classifier.
* bert.ipynb uses the BERT model
* distilbert.ipynb uses the DistilBERT model
* bayesian.ipynb uses the Bayesian model
* random_forest.ipynb uses the Random Forest model
* cnn.ipynb uses a convoltuional neural network model

## Project Instructions

### Goals and the Task

The goal of this assignment is to design and implement a classifier using the pre-trained BERT model to identify from a set of books which were writeen by G. K. Chesterton. Additional models were tested to compare how much better or worse they perform compared to BERT for correct authorship attribution. 

### Steps

#### Dataset Compiling

* Novels in text format from Project Guttenberg downloaded and prepared for analysis (normalization and tokenization)
* Use stories of the same genre (i.e. murder mystery)

#### Feature Engineering

* Embeddings used from the BERT model for all feature extraction

#### Statistical Modeling:

Different machine learning models were analyzed to produce a binary classification of stories that either were or were not written by G. K. Chesteron
* BERT
* DistilBERT
* Bayesian
* Random Forest
* Convolutional Neural Network

#### Evaluation

* Classifier accuracies collected and compared to determine success of the implementation
