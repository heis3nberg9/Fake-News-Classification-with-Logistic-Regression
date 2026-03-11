# Fake News Classification with Logistic Regression
##  Overview
We can't trust everything we come across online. All news are not real, right?
So how will you detect fake news?
The Answer is Python/ML. In this project we build a machine learning model to classify news articles as FAKE or REAL. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a LogisticRegression Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.


## Dataset
* **Source file:** https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view
* **Size:** 6,335 articles
* **Columns:** title, text, label (FAKE / REAL)
* The dataset is fairly balanced between FAKE and REAL articles, as shown in the label distribution plot.


## Approach
* **Preprocessing:**
Combined the title and text columns into a single feature to give the model more signal.
Split the data into 80% training and 20% testing using stratified sampling to preserve class balance.

* **Feature Extraction:** TF-IDF Vectorization
Used TfidfVectorizer with English stop words removed and a max_df=0.7 threshold to filter out terms that appear in more than 70% of documents.
fit_transform was applied only on the training set; the test set was only transformed — this prevents data leakage.

* **Model:**  Logistic Regression
Trained a LogisticRegression model with max_iter=1000 to ensure convergence.


## Results
| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 95.66% |
| **Testing Accuracy** | 92.03% |


## Confusion Matrix (Test Set):
| Predicted REAL | Predicted FAKE |
| :--- | :--- |
Actual REAL        | 600    |         33|
Actual FAKE         | 68      |        566|

True Positives (REAL correctly identified): 600
True Negatives (FAKE correctly identified): 566
False Positives (FAKE misclassified as REAL): 68
False Negatives (REAL misclassified as FAKE): 33
