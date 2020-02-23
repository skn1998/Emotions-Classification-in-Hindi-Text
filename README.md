# Emotions-Classification-in-Hindi-Text
Codes and Data Set for Emotions Classification in Hindi Text

EMOTIONS CLASSIFICATION IN HINDI TEXT

## Steps followed to built the model:
* Data has been read into lists and have been finally converted into a Pandas DataFrame.
* Input to the model (X) and output (y) has been separated.
* Using CountVectorizer text has been converted into numerics as model needs numbers not String or tokens. Here we are giving it some stop words so that they can be removed as they do not contribute to the model building. Also a tokenizer defined manually has been passed as the built-in tokenizer will split everything into characters which we don’t want so the manually built tokenizer (my_tokenizer) splits about white spaces only. Also  we are using n_grams to preserve some local meanings.
* A logistic regression model has been trained as Logistic Regression works well for the Sparse Matrices.
* Train Accuracy and Test Accuracy has been evaluated.
* After that I have trained the model on the whole data Set and have shown cross validation score in each case as well as the overall cross validation score.


## Obtained Results:
* Train Accuracy: 0.76
* Test Accuracy: 0.65
* Cross Validation Accuracy: 0.61


## Analysis of the results:
* Clearly the model is a bit overfitting as there is a small difference between Train and Test Accuracy.
* Also since we have very little data (considering the Hindi aspect where we have too many characters and other things) it is difficult to increase the accuracy. However the trained model has been able to catch some important words for the purpose of classification which can be seen in this diagram taken from the Jupyter Notebook.
* Stop words are only available for a few languages. For Hindi we don't have inbuilt stop words. So we need to provide stop words on our own.
