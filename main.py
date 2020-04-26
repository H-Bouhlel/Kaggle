import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence

from sklearn.metrics import accuracy_score
from preprocessing import preprocessing, tokenize_fct
from model import model


df_train = pd.read_csv(r"C:\Users\ahmed\Desktop\M2_BD\2esemeste\Projet_kaggle\Semtiment_analysis\train.csv")
df_test = pd.read_csv(r"C:\Users\ahmed\Desktop\M2_BD\2esemeste\Projet_kaggle\Semtiment_analysis\test.csv")

df_train.replace({"sentiment":{"positive":2, "neutral":1, "negative":0}}, inplace=True)
df_test.replace({"sentiment":{"positive":2, "neutral":1, "negative":0}}, inplace=True)

df_test["text"]=preprocessing(df_test,"text")
df_train["text"]=preprocessing(df_train,"text")

X_train = df_train.text
y_test=df_test.sentiment
y_train = df_train.sentiment

tokenize=tokenize_fct(X_train)

X_test = df_test.text

X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)

max_len= max([len(s.split()) for s in df_train['text']])

X_train = pad_sequences(X_train, max_len)
X_test = pad_sequences(X_test, max_len)

word_len = len(tokenize.word_index)+1

model=model(word_len,max_len)
print(model.summary)

model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
final_pred = model.predict_classes(X_test)

accuracy_score(final_pred,y_test)



