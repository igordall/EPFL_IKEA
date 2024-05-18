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

Our journey began with an assessment of how traditional machine learning methods performed on our dataset (Logistic Regression, KNN, Decision Tree, and Random Forest). After evaluating these initial approaches, we transitioned to using the state-of-the-art BERT model to enhance our predictive capabilities. The table below provides an overview of the performance of each model based on key metrics. 


|                  | Logistic Regression | KNN     | Decision Tree | Random Forest | Camembert |
|------------------|---------------------|---------|---------------|---------------|-----------|
| Precision        |         0.44        |   0.29  |      0.31     |      0.37     |      -    |
| Recall           |         0.45        |   0.19  |      0.31     |      0.39     |      -    |
| F1-score         |         0.44        |   0.11  |      0.31     |      0.37     |      -    |
| Accuracy         |         0.45        |   0.20  |      0.31     |      0.35     |      -    |


Understanding the differences in these metrics is crucial for comprehending model performance. **Precision** quantifies the accuracy of positive <u>predictions</u>, highlighting the proportion of true positives among all positive predictions. **Recall** assesses the model's ability to identify all relevant instances, focusing on the proportion of actual positives correctly identified. **F1-score** provides a balance between precision and recall, useful in situations with uneven class distributions. **Accuracy** reflects the overall correctness of the model across all predictions, relevant when the classes are symmetric in size and the costs of false positives and false negatives are similar.

In our case, the respective subsections are more or less even, suggesting balanced model performance in terms of handling different classes and types of errors. In subsequent sections, we will delve deeper into each model, exploring their advantages and limitations in detail.

### Logistic Regression

We selected Logistic Regression as our initial approach to explore how machine learning can predict sentence difficulties. This model is computationally efficient, enabling rapid training and prediction times. It operates by vectorizing sentences into a numerical form, which the machine learning algorithm utilizes to perform regression. A significant advantage of Logistic Regression is its interpretability; it directly reveals the influence of each predictor (regression coefficient) on the sentence difficulty.

We divided the training data on an 80/20 basis, using 80% for training the model and the remaining 20% for testing. The results presented in the table above reflect this division. Although the regression model performs slightly better than other traditional models, it only exhibits moderate capability in accurately identifying sentence difficulties, with Precision and Recall values of 44% and 45%, respectively. This indicates that the model's predictions are correct 44% of the time when it predicts difficulty, and it successfully identifies 45% of all actual difficulties.

To better understand potential errors, we will examine the confusion matrix for an overview and then delve into specific misclassification. The confusion matrix, based on predicted and actual values, indicates a pronounced tendency to misclassify sentences in the middle difficulty range (B2 to C1). These sentences likely share more overlapping features with neighboring classes compared to those at the extremes, which tend to be more distinct.

<img width="600" alt="Capture d’écran 2024-05-18 à 16 56 37" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/a8df770f-2c11-4a0e-b3fa-660cbdb4f858">

For a more detailed analysis, consider the mispredicted sentence "Un service précieux qui a sans doute permis de sauver des vies," which was erroneously classified as C1 when its actual difficulty level is A2. An examination of the vectorization and predictors reveals that the five most influential features leading to a C1 prediction are common determiners and prepositions such as "des," "sa," "les," "de," and "du" (equivalent to English "the," "those," or "some"). These words, typically neutral regarding difficulty level, suggest the model may not adequately capture the complexity of technical vocabulary or sentence structure, such as technical vocabulary, grammar or punctuation. However, the model accurately predicts that longer sentences are typically more difficult, as demonstrated by the correctly C1 classified sentence "En attribuant une valeur de flux potentiel aux liens du graphe, la segmentation fondée sur le critère de modularité aboutit à un découpage considéré comme optimal du point de vue de la dynamique démographique interne à chaque compartiment." 

Those limitations suggest further refinement is needed to enhance the model’s ability to differentiate between linguistic complexity and common vocabulary. You can find the used code below: 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = df_training.copy()

# Divide data into features (X) and target (y)
X = df['sentence']
y = df['difficulty']

# Convert difficulty labels into numeric values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing set (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a numeric representation of sentences using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for imbalanced classes
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)

```

### KNN

We also used a k-Nearest Neighbors (kNN) algorithm to tackle the difficulty prediction. The principle behind kNN is to find a predefined number of training samples closest in distance to the new point and predict the label from these, relying on features from sentences that are believed to contribute to their difficulty. To compute the distance the algorithm will measure the similarity between sentences using sentences with similar lengths and complexities.

Again using the technique 80/20, we got the results mentioned in the initial table. The outcome suggests poor prediction with a Precision of 29% but still better than identification capabilities with a Recall of 19%, resulting in more false positives than false negatives.  

To better understand the reason why the model fails to properly assess the difficulty, visioning the confusion matrix helps us to identify that the vast majority of predictions go for A1 resulting in a lot of false positives. 

<img width="600" alt="Capture d’écran 2024-05-18 à 19 12 59" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/ca1916f7-1333-440d-a50d-28909fa3c0e5">



### Decision Tree

### Random Forest

### Camembert















