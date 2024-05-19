# EPFL_IKEA: AI-Powered French Text Difficulty Prediction

## Project Overview

In this project, Alex and I harness artificial intelligence to predict the difficulty levels of French texts. Our experience as native French speakers who have learned English informs our appreciation for matching language learners with texts of appropriate complexity. This GitHub repository documents our journey from participating in a Kaggle competition to developing a model that leverages our fluency to benefit language learners worldwide.

## Data Resources

We used the following datasets for training and evaluating our model:

- **Training Data:** [Training Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/training_data.csv)
- **Unlabelled Test Data:** [Test Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/unlabelled_test_data.csv)
- **Sample Submission:** [Sample Submission File](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/sample_submission.csv)
- **Additional Sentences:** [French Sentences](https://raw.githubusercontent.com/AlexPinel06/Team-IKEA-ML/main/data/french_sentences_realistic.csv)

## Model Development and Performance

### Initial Machine Learning Approaches

Our journey began by testing traditional machine learning methods:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

### Advanced Modeling with BERT

After initial experiments, we integrated the BERT model to enhance our prediction capabilities, focusing on its advanced contextual understanding of language.

### Performance Metrics Overview

We assess our models using key metrics to ensure precision, recall, F1-score, and accuracy. Here's how each model performed:


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

<img width="500" alt="Capture d’écran 2024-05-18 à 16 56 37" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/a8df770f-2c11-4a0e-b3fa-660cbdb4f858">

For a more detailed analysis, consider the mispredicted sentence "Un service précieux qui a sans doute permis de sauver des vies," which was erroneously classified as C1 when its actual difficulty level is A2. An examination of the vectorization and predictors reveals that the five most influential features leading to a C1 prediction are common determiners and prepositions such as "des," "sa," "les," "de," and "du" (equivalent to English "the," "those," or "some"). These words, typically neutral regarding difficulty level, suggest the model may not adequately capture the complexity of technical vocabulary, grammar, or punctuation. However, the model accurately predicts that longer sentences are typically more difficult, as demonstrated by the correctly C1 classified sentence "En attribuant une valeur de flux potentiel aux liens du graphe, la segmentation fondée sur le critère de modularité aboutit à un découpage considéré comme optimal du point de vue de la dynamique démographique interne à chaque compartiment." 

Those limitations suggest further refinement is needed to enhance the model’s ability to differentiate between linguistic complexity and common vocabulary. You can find the used code below: 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = df_training.copy()

X = df['sentence']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

metrics = precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
print(f"Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}, Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

### KNN

We then tackled classification algorithms to predict difficulty, methods that assign predefined categories or labels to new instances based on their features. Indeed, the kNN algorithm works by finding a set number of training examples that are closest to a new data point and using their labels to make a prediction. It calculates how close each sentence is to others based, for instance, on similar sentence lengths and complexities.

Using the 80/20 technique again, we obtained the results presented in the initial table. The outcome suggests poor prediction performance, with a Precision of 29%. However, the identification capabilities are even lower, with a Recall of 19%, resulting in more false positives than false negatives.

To better understand why the model fails to properly assess difficulty, examining the confusion matrix helps. It reveals that the vast majority of predictions are for A1, resulting in many false positives. This suggests that the model relies heavily on common words to cluster difficulty labels, which do not carry much discriminative information about the complexity or difficulty of a sentence.

<img width="500" alt="Capture d’écran 2024-05-18 à 19 12 59" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/ca1916f7-1333-440d-a50d-28909fa3c0e5">

To dive deeper, we plotted the most frequent words in misclassified sentences, which include words like "de," "la," "et," and "les." This shows that our model is overly influenced by common French articles, prepositions, and conjunctions, leading to clustering around the A1 prediction. For instance, in the following misclassified C1 sentence as A1, there are several occurrences of "de," "la," and "et": "La raison de cette ambivalence précède l'existence des robots et même leur nom: elle est culturelle et se cache dans le vieux mythe du Golem remis à l'honneur en Occident par Frankenstein de Mary Shelley en 1818."

To prevent this issue removing common stop words during the vectorization process would help the model focus on more meaningful features. You can find the code used in the following lines : 

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df_KNN = df_training.copy()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_KNN['sentence'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_KNN['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

metrics = precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}, Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}")
```

### Decision Tree

Another classification algorithm is the decision tree, which operates by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. The model is structured as a tree where each internal node represents a "test" on an attribute (in our case, whether a word or a sentence), each branch represents the outcome of the test, and each leaf node represents a class label (in our case, the difficulty).  

Again using the 80/20 technique yielded the results from the table above, suggesting an even distribution among the classes and types of errors. 

<img width="500" alt="Capture d’écran 2024-05-19 à 10 31 28" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/4c4e5dee-f5c1-4925-a2ea-53558e75a6c2">

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load and prepare the dataset
df_DT = df_training.copy()  # Assuming df_training is predefined
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_DT['sentence'])  # Convert text to word frequency vectors
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_DT['difficulty'])  # Encode labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict and calculate metrics
y_pred = dt_classifier.predict(X_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")
```
### Random Forest

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df_RF = df_training.copy()  # Load your data

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_RF['sentence'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_RF['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}, Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```
### Camembert















