# EPFL_IKEA: AI-Powered French Text Difficulty Prediction

## Project Overview

In this project, Alex and I harness artificial intelligence to predict the difficulty levels of French texts. Our experience as native French speakers who have learned English informs our appreciation for matching language learners with texts of appropriate complexity. This GitHub repository documents our journey from participating in a Kaggle competition to developing a model that leverages our fluency to benefit language learners worldwide.

## Data Resources

We used the following datasets for training and evaluating our model:

- **Training Data:** [Training Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/training_data.csv)
- **Unlabelled Test Data:** [Test Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/unlabelled_test_data.csv)
- **Sample Submission:** [Sample Submission File](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/sample_submission.csv)
- **Additional Sentences:** [French Sentences](https://raw.githubusercontent.com/AlexPinel06/Team-IKEA-ML/main/data/french_sentences_realistic.csv)
- Description of the data ...

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
| Recall           |         0.45        |   0.20  |      0.31     |      0.39     |      -    |
| F1-score         |         0.44        |   0.11  |      0.31     |      0.37     |      -    |
| Accuracy         |         0.45        |   0.20  |      0.31     |      0.35     |      -    |


Understanding the differences in these metrics is crucial for comprehending model performance. **Precision** quantifies the accuracy of positive <u>predictions</u>, highlighting the proportion of true positives among all positive predictions. **Recall** assesses the model's ability to identify all relevant instances, focusing on the proportion of actual positives correctly identified. **F1-score** balances precision and recall, useful in situations with uneven class distributions. **Accuracy** reflects the overall correctness of the model across all predictions, relevant when the classes are symmetric in size and the costs of false positives and false negatives are similar.

In our case, the respective subsections are more or less even, suggesting balanced model performance in terms of handling different classes and types of errors. In subsequent sections, we will explore each model's advantages and limitations in detail.

### Logistic Regression

We selected Logistic Regression as our initial method to explore predicting sentence difficulties using machine learning. This model stands out for its computational efficiency, facilitating swift training and prediction processes. Logistic Regression operates by first converting sentences into numerical form—specifically, by creating features based on the characteristics of the sentences, such as word frequency, sentence length, and syntax complexity. The model then applies these numerical features to a logistic function, which estimates the probability that a sentence belongs to a particular difficulty category. A significant strength of Logistic Regression is its interpretability; it provides clear insights by showing how each feature’s weight (regression coefficient) influences the predicted difficulty level, allowing for easier adjustments and understanding of the model's decisions.

We split the training data using an 80/20 ratio, with 80% used for training the model and the remaining 20% for testing. The results, as shown in the table above, reflect this setup. The regression model slightly outperforms other traditional models, but it only moderately succeeds in accurately determining sentence difficulties, achieving Precision and Recall rates of 44% and 45%, respectively. This means the model correctly predicts sentence difficulty 44% of the time and accurately identifies 45% of all actual difficult sentences.

To delve deeper into where the model may be going wrong, we analyzed the confusion matrix, which highlights the model's tendency to misclassify sentences, particularly those in the middle difficulty range (B2 to C1). These sentences likely have features that overlap significantly with adjacent difficulty levels, unlike sentences at the extremes, which tend to have more distinct characteristics.

<img width="500" alt="Capture d’écran 2024-05-18 à 16 56 37" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/a8df770f-2c11-4a0e-b3fa-660cbdb4f858">

A case in point is the misclassified sentence "Un service précieux qui a sans doute permis de sauver des vies," incorrectly labeled as C1 instead of its actual difficulty of A2. Analysis of its vectorization and contributing predictors shows that the most influential features leading to the incorrect C1 classification were common determiners and prepositions like 'des', 'sa', 'les', 'de', and 'du'. These frequently used words are typically neutral concerning difficulty and indicate that the model might struggle with capturing the nuances of technical vocabulary, grammar, or punctuation. Notably, attempts to improve model performance by removing these "stop words" resulted in poorer outcomes with an accuracy of 42%, suggesting that common words still hold significant predictive value. Nevertheless, the model does recognize that longer sentences generally pose more difficulty, as evidenced by correctly classifying the complex sentence: "En attribuant une valeur de flux potentiel aux liens du graphe, la segmentation fondée sur le critère de modularité aboutit à un découpage considéré comme optimal du point de vue de la dynamique démographique interne à chaque compartiment."

These insights highlight the need for further refinement to better distinguish between linguistic complexity and commonplace vocabulary in our model. Below is the code we used for this analysis:

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

Following our use of Logistic Regression, we explored another approach using the k-Nearest Neighbors (kNN) classification algorithm to predict sentence difficulties. This method categorizes sentences by assigning them to predefined difficulty levels based on their features. Specifically, kNN operates by identifying a specified number of training examples that are closest to a new sentence in terms of feature similarity—such as sentence length and complexity. It then predicts the difficulty level of the new sentence based on the most common categories among these nearest neighbors. This method relies heavily on the assumption that similar sentences share similar difficulty levels, making it particularly effective when clear patterns of similarity exist within the data.

Using the 80/20 split technique again, we obtained the results presented in the initial table. The outcome suggests poor prediction performance, with a Precision of 29%. However, the identification capabilities are even lower, with a Recall of 20%, resulting in more false positives than false negatives.

Examining the confusion matrix provides insights into why the model fails to properly assess difficulty. It reveals that the vast majority of predictions are for A1, resulting in many false positives. The model’s performance might be hindered by the high dimensionality of text data, which results from the large number of features derived from each word. This complexity makes it more challenging to identify clear patterns and accurately find nearest neighbors. This stresses the poor scalability of the kNN model.

<img width="500" alt="Capture d’écran 2024-05-18 à 19 12 59" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/ca1916f7-1333-440d-a50d-28909fa3c0e5">

To delve deeper, we analyzed the most frequent words in misclassified sentences (stop words excluded while counting), including words like 'plus', 'comme', 'cette', and 'si', even after excluding stop words into the model—which, once more, resulted in a poorer outcome. This shows that our model is overly influenced by common French adverb, adjective, and conjunctions, leading to clustering around the A1 prediction. Indeed, the concept of the "nearest" neighbor becomes less meaningful. Distances between points (words/ sentences) can become uniformly large or not sufficiently distinct to discern close from distant neighbors effectively. For instance, the sentence B2 was wrongly classified as A2, "Si cette dernière innovation voyait le jour, une de ses applications serait tout aussi impressionnante et inquiétante." contains occurrences of 'cette', and 'si'.

These results demonstrate that kNN is not well-suited for handling the complexities of written text. Below is the code used for this analysis:
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

We further expanded our exploration of machine learning models for predicting sentence difficulties by implementing a Decision Tree algorithm. This classification method predicts the difficulty level of sentences by constructing a model that learns simple decision rules from the data features. In our application, the model is visualized as a tree structure: each internal node represents a "test" related to an attribute of the sentence, such as the presence of specific words or the complexity of the sentence structure. Each branch stemming from these nodes denotes the outcome of the test, and each leaf node corresponds to a class label, which in this case is the predicted difficulty level. This hierarchical approach allows the Decision Tree to make clear and logical decisions based on the hierarchical significance of sentence features, making it both interpretable and effective for tasks with well-defined rules and distinctions between categories.

Again using the 80/20 technique yielded the results from the table above, suggesting an even distribution among the classes and types of errors. The model once more produces poor predicting results, having an accuracy of 31%. 

Diving into the confusion matrix, we spot that the model struggles to point out difficulty subtilities, having a lot of false positives and negatives. This suggests an overfitting issue where the model captures noise rather than actual difficulty boundaries.   

<img width="500" alt="Capture d’écran 2024-05-19 à 10 31 28" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/4c4e5dee-f5c1-4925-a2ea-53558e75a6c2">

Analyzing the erroneous prediction closely, we plotted the top 10 important features resulting in higher decision power. It resulted again in common French words including 'de', 'la', 'les', and 'et' with, once more, detrimental model results if excluding stop words. This shows that our model is overly influenced by common French articles, prepositions, and conjunctions, leading to misreading true language difficulties as distinctions are not well-defined. Indeed, the algorithm struggles with noisy labels and irrelevant features, such as articles and propositions, which complicates difficulty prediction. Moreover, as the model overestimates common words small variations in the data might result in a completely different tree being generated. 

Those issues stress again the limitations of such classification algorithms to tackle our predicting projects. You can find the following code studied in our analysis:  
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

Continuing with our exploration of machine learning techniques for predicting sentence difficulties, we implemented the Random Forest algorithm. This advanced classification method extends the concept of decision trees by creating an ensemble of decision trees trained on different subsets of the data and using different subsets of features at each split within those trees. Each tree in the forest makes its own prediction, and the final output is determined by the majority vote across all trees. This approach not only enhances the predictive accuracy by reducing the risk of overfitting associated with single decision trees but also maintains good interpretability, as it allows for an analysis of feature importance across multiple trees. Random Forest is particularly robust and effective in dealing with both large datasets and datasets with a high dimensionality of features, making it well-suited for complex classification tasks like predicting the difficulty of sentences.





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















