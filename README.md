# EPFL_IKEA


## Background 

In this GitHub project, Alex and I aim to predict the difficulty level of French texts using artificial intelligence. As native French speakers who have learned English, we understand the importance of engaging with texts that match our comprehension level. Learning with appropriately challenging texts can significantly enhance the learning process by introducing increasingly difficult material at a manageable pace.

Prompted by these insights, we participated in a Kaggle competition to find the most effective prediction model. This competition inspired us to develop a model that not only aids in language learning but also leverages our native fluency in French. Our goal is to determine the difficulty level of unfamiliar texts using the Common European Framework of Reference for Languages (CEFR) scale from A1 to C2. To achieve this, we have a training dataset comprising 4,800 French sentences, each annotated with its actual difficulty level. Additionally, we will evaluate our model’s accuracy using an unlabeled dataset.

## Data Used 
sample submission: https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/sample_submission.csv

training data: https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/training_data.csv

unlabelled test data: https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/unlabelled_test_data.csv

Fr_sen : https://raw.githubusercontent.com/AlexPinel06/Team-IKEA-ML/main/data/french_sentences_realistic.csv

## Model used

Our journey began with an assessment of how traditional machine learning methods performed on our dataset. After evaluating these initial approaches, we transitioned to using the state-of-the-art BERT model to enhance our predictive capabilities. The table below provides an overview of the performance of each model based on key metrics. 


|                  | Logistic Regression | KNN     | Decision Tree | Random Forest | Camembert |
|------------------|---------------------|---------|---------------|---------------|-----------|
| Precision        |         0.44        |   0.29  |      0.31     |      0.37     |      -    |
| Recall           |         0.45        |   0.19  |      0.31     |      0.39     |      -    |
| F1-score         |         0.44        |   0.11  |      0.31     |      0.37     |      -    |
| Accuracy         |         0.45        |   0.20  |      0.31     |      0.35     |      -    |


Understanding the differences in these metrics is crucial for comprehending model performance. **Precision** quantifies the accuracy of positive predictions, highlighting the proportion of true positives among all positive predictions. **Recall** assesses the model's ability to identify all relevant instances, focusing on the proportion of actual positives correctly identified. **F1-score** provides a balance between precision and recall, useful in situations with uneven class distributions. **Accuracy** reflects the overall correctness of the model across all predictions, relevant when the classes are symmetric in size and the costs of false positives and false negatives are similar.

In our case, the respective subsections are more or less even, suggesting balanced model performance in terms of handling different classes and types of errors. In subsequent sections, we will delve deeper into each model, exploring their advantages and limitations in detail.

### Logistic Regression

We opted for the Logistic Regression to grasp the first insight into how ML could predict sentence difficulties. We split the training data into an 80/20 scale, training our model on 80% and testing on the remaining 20% which gave us the score in the table above. 

### KNN

### Decision Tree

### Random Forest

### Camembert















