# Fake News Detection
In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python.
## Prerequisites
-Python 3.6
-Sklearn (scikit-learn)
-numpy
-scipy
## Dataset
The LIAR dataset was used for this project, and it consists of three.tsv files for test, train, and validation. It contains 13 columns for train, test and validation sets.
LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION
Two columns are selected to use in this project:
-Column 1: Statement (News headline or text).
-Column 2: Label (Label class contains: True, False)
The dataset used in this project is in the .csv format, the original .tsv files can be found in liar_dataset.
## Steps to run
Step 1: Clone this github repository and set it as your working directory by the following command:
!git clone https://github.com/dharace/Fake-News-Detection.git
!cd /content/Fake-News-Detection
Step 2: Install all the dependencies from the requirements.txt
pip install -r requirements.txt
Preview of the code can be accessed through this [ipynb](https://colab.research.google.com/github/dharace/Fake-News-Detection/blob/main/TestFakeNewsDetection.ipynb#scrollTo=Dc3QFmjhCfF6) notebook.
At the end of the program, you will be asked for an input which will be a piece of information or a news headline that you want to verify. Once you paste or type news headline, then press enter.
The output of the true or false news will be produced along with the probability.
