
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer


def preprocessing (df,column) :
    df[column]=df[column].str.lower()
    df[column]=df[column].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',str(x))))
    return df[column]   

def tokenize_fct (data):
    tokenize = Tokenizer()
    tokenize.fit_on_texts(data.values)
    return tokenize
