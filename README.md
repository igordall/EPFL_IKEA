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


Understanding the differences in these metrics is crucial for comprehending model performance. **Precision** quantifies the accuracy of positive predictions, highlighting the proportion of true positives among all positive predictions. **Recall** assesses the model's ability to identify all relevant instances, focusing on the proportion of actual positives correctly identified. **F1-score** provides a balance between precision and recall, useful in situations with uneven class distributions. **Accuracy** reflects the overall correctness of the model across all predictions, relevant when the classes are symmetric in size and the costs of false positives and false negatives are similar.

In our case, the respective subsections are more or less even, suggesting balanced model performance in terms of handling different classes and types of errors. In subsequent sections, we will delve deeper into each model, exploring their advantages and limitations in detail.

### Logistic Regression

We opted for the Logistic Regression to grasp the first insight into how ML could predict sentence difficulties. Indeed the model is computationally inexpensive, allowing for quick model training and prediction. The model works by vectorizing the sentences into a numeric form that can be used by the machine learning algorithm to run a regression. The added value of the model is interpretability, giving direct insights into how much each predictor (coefficient) affects the target variable (difficulty). 

We split the training data into an 80/20 scale, training our model on 80%, and testing on the remaining 20% which gave us the score in the table above. Regarding the other model traditional model performs well, however suggesting a moderate ability to capture the true instances of sentence difficulties, recognizing only 45% of them based on the Recall.

Diving into the reasoning behind the potential mistakes, we will first have a look at the confusion matrix to have an overall understanding then dive into specific mistakes. First, regarding the confusion matrix we spot a higher tendency to misclassify in the middle-range value from B2 to C1, as they must have more overlapping features with their neighbors then the extreme range that are likely more distinct.  

<img width="462" alt="Capture d’écran 2024-05-18 à 16 56 37" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/a8df770f-2c11-4a0e-b3fa-660cbdb4f858">

Second, let's have a closer look at some erroneous predictions. For instance, the sentence "Un service précieux qui a sans doute permis de sauver des vies" has been predicted C1 where its actual difficulty is A2, looking at the vectorization and the predictor, we spot that the 5 most important features influencing the prediction to C1 are word such as "des, sa, les, de, du" which are determiners and prepositions such as "the, those" or "some", which those not actully impact the real level of a sentence because it would be located in easy and hard sentence. This stresses that the model struggles to really identify the driver of the difficulties in sentences such as technical lexical frame or punctuation. However, the model seems to manage properly that the longer the sentence is the more difficult it is, such at this one that has been properly predicted C1 "En attribuant une valeur de flux potentiel aux liens du graphe, la segmentation fondée sur le critère de modularité aboutit à un découpage considéré comme optimal du point de vue de la dynamique démographique interne à chaque compartiment."



### KNN

### Decision Tree

### Random Forest

### Camembert















