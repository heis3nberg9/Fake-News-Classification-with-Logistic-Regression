# Fake News Classification System

A Machine Learning project to detect misinformation using Logistic Regression and NLP.

🔗 Live Demo
Check out the live web application here:

👉 (https://fake-news-detector-with-logistic-regression.streamlit.app/)

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heis3nberg9/Fake-News-Classification-with-Logistic-Regression/blob/main/notebooks/fake_news_classification.ipynb)

This project achieves 92% accuracy using Logistic Regression...

##  Overview
We can't trust everything we come across online. All news are not real, right?
So how will you detect fake news?
The Answer is Python/ML. In this project we build a machine learning model to classify news articles as FAKE or REAL. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a LogisticRegression Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.


## Dataset
* **Source file:** https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view
* **Size:** 6,335 articles
* **Columns:** title, text, label (FAKE / REAL)
* The dataset is fairly balanced between FAKE and REAL articles, as shown in the label distribution plot.


## Technical Stack
* Language: Python
* Libraries: Scikit-learn, Pandas, NumPy, Streamlit
* Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)
* Classifier: Logistic Regression
* Environment: WSL2 (Ubuntu)

## Features
* **Instant Prediction:** Users can input any headline and get an immediate classification.

* **Interactive Web UI:** Built with Streamlit for a seamless user experience.


## Results
| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 95.66% |
| **Testing Accuracy** | 92.03% |


## Confusion Matrix (Test Set):
| | Predicted REAL | Predicted FAKE |
|---|---|---|
| Actual REAL | 600 | 33 |
| Actual FAKE | 68 | 566 |

* True Positives (REAL correctly identified): 600
* True Negatives (FAKE correctly identified): 566
* False Positives (FAKE misclassified as REAL): 68
* False Negatives (REAL misclassified as FAKE): 33

## Future Improvements

* Integration of Sentiment Analysis to detect emotional manipulation.

## Summary
Fake news spreads rapidly online and can have serious real-world consequences — from misleading the public to fueling panic and misinformation. Manually fact-checking every article is impractical, which is where Machine Learning comes in.
This project builds a fake news classifier using Python and Natural Language Processing (NLP). The model is trained on a labeled dataset of real and fake news articles, learning to detect patterns in the text that distinguish credible reporting from misinformation. Given a news headline or article, it predicts whether the content is REAL or FAKE.
The pipeline uses TF-IDF vectorization to convert raw text into meaningful numeric features, and Logistic Regression as the classification model achieving a test accuracy of 92%
