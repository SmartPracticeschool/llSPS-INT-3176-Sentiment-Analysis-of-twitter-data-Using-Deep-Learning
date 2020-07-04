# Importing libraries
import numpy as np
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('train.tsv', sep="\t")

#dataset2 = dataset

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]

def remove_pat(inpt, pat): #for removing the @names and links in the comments
    n = re.findall(pat, inpt)
    for i in n:
        inpt = re.sub(i, ' ', inpt)
    return inpt

for i in range(0, 7589):
    rev = dataset['tweet'][i]
    rev = remove_pat(rev, '@[\w]*') #user name
    rev = rev.replace('(', '') #bracket one
    rev = rev.replace(')', '') #bracket two
    rev = remove_pat(rev, r'https?://[A-Za-z0-9./]+')
    #rev = remove_pat(rev, r"http\S+") #links
    rev = re.sub('[^a-zA-Z]', ' ',rev) #removing special characters
    rev = rev.lower() #lower case
    rev = rev.split()
    rev = [word for word in rev if not word in set(stopwords.words('english'))] #getting rid of stopwords
    ps = PorterStemmer()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))] #Stemming
    rev = ' '.join(rev)
    c.append(rev)

#dataset2['tweet'] = c

# Tokenizing for words into sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(c)
x = tokenizer.texts_to_sequences(c)
x = pad_sequences(x)

# OneHotEncoding the target
y = pd.get_dummies(dataset['label']).values

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

# Model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SpatialDropout1D
model = Sequential()
model.add(Embedding(3000, 200,input_length = 101))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 4, batch_size = 32)

y_pred = model.predict(x_test)

# Prediction
def prediction(text):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=101)
    score = model.predict([x_test])[0]
    if np.argmax(score) == 2:
        a = "POSITIVE"
    elif np.argmax(score) == 0:
        a = "NEGATIVE"
    elif np.argmax(score) == 1:
        a = "NEUTRAL"
    return print(a)

prediction('I hate mondays.')

# Saving
model.save('nlp2.h5')