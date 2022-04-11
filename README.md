# Fake News Detection
In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python.

## Prerequisites

-Python 3.6</br>
-Sklearn (scikit-learn)</br>
-numpy</br>
-scipy</br>
## Dataset
The LIAR dataset was used for this project, and it consists of three.tsv files for test, train, and validation. It contains 13 columns for train, test and validation sets.

LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

Two columns are selected to use in this project:

 -Column 1: Statement (News headline or text).</br> 
 -Column 2: Label (Label class contains: True, False)
 
The dataset used in this project is in the .csv format, the original .tsv files can be found in liar_dataset.
## Steps to run
Step 1: Clone this github repository and set it as your working directory by the following command:
```
!git clone https://github.com/dharace/Fake-News-Detection.git
!cd /content/Fake-News-Detection
```
Step 2: Install all the dependencies from the Requirements.txt
```
pip install -r Requirements.txt
```
Step 3: Import required classes from github to call methods.</br>
Step 4: Install dataset.</br>
Step 5: Go to the "data" folder & download glove.6B.zip.</br>
Step 6: Create object of DataPrep.py and pass filepath of each dataset test.csv, train.csv, valid.csv.</br>
Step 7: Create object of FeatureSelection.py and pass filepath of glove_6B_50d.</br>
Step 8: Create object of Classifier.py and pass CountVectorizer object, test & train dataset, tfidf_ngram and final model.</br>
Step 9: Test the result.</br>
Preview of the code can be accessed through this [ipynb](https://colab.research.google.com/github/dharace/Fake-News-Detection/blob/main/TestFakeNewsDetection.ipynb#scrollTo=Dc3QFmjhCfF6) notebook.
At the end of the program, you will be asked for an input which will be a piece of information or a news headline that you want to verify. Once you paste or type news headline, then press enter.
The output of the true or false news will be produced along with the probability.

## File Description
The files mentioned in src folders are as follows:</br>
### DataPrep.py
This file contains all of the procedures required to process all of the input documents and texts. We read the train, test, and validation data files first, then did some pre-processing such as tokenizing and stemming. Some exploratory data analysis is carried out, such as response variable distribution and data quality checks such as null or missing values, among other things.
### FeatureSelection.py
We used feature extraction and selection algorithms from the sci-kit learn python libraries in this file. We utilised simple bag-of-words and n-grams for feature selection, and then tf-tdf weighting for term frequency. We've also utilised word2vec and POS tagging to extract the features, albeit POS tagging and word2vec aren't being used right now.
### classifier.py
In this section, we have constructed all the classifiers for predicting fake news detection. Each classifier uses the features that were extracted. Stochastic gradient descent, Naive-bayes, Logistic Regression, Linear SVM, and Random forest classifiers from Sklearn were used. Feature extractions were all incorporated into the classifiers. The model was fitted, the f1 score was compared, and the confusion matrix was checked. Based on the fitting of all the classifiers, two of the best performing models were selected as candidates. GridSearchCV methods have been implemented on these candidate models for parameter tuning, and we selected the best performing parameters for these classifiers. Finally, a model was selected to detect fake news with probability of truth. In Addition to this, We have also extracted the top 50 features from our term-frequency tfidf vectorizer to see what words are most and important in each of the classes. Furthermore, we have used Precision-Recall and learning curves to quantify how training and test sets perform when we increase the amount of data in our classifiers.
### prediction.py
Once you close this repository, this model will be copied to user's machine and will be used by prediction.py file to classify the fake news. The user inputs a news article and the model calculates the final classification output and probability of truth that is exhibited to the user.
