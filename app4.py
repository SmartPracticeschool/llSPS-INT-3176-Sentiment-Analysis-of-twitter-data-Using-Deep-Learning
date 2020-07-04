
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
graph = ops.get_default_graph()


from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    s = str(name)
    with graph.as_default():
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np
        model = load_model('nlp2.h5')
        tokenizer = Tokenizer(num_words=3000)
        x_test = pad_sequences(tokenizer.texts_to_sequences([s]), maxlen=101)
        score = model.predict([x_test])[0]
        if np.argmax(score) == 2:
            a = "POSITIVE"
        elif np.argmax(score) == 0:
            a = "NEGATIVE"
        elif np.argmax(score) == 1:
            a = "NEGATIVE"
        
    print(s)
    return render_template('index.html', abc = a)

def prediction(text):
    tokenizer = Tokenizer(num_words=3000)
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=101)
    score = model.predict([x_test])[0]
    if np.argmax(score) == 2:
        a = "POSITIVE"
    elif np.argmax(score) == 0:
        a = "NEGATIVE"
    elif np.argmax(score) == 1:
        a = "NEUTRAL"
    return print(a)


if __name__ == '__main__':
    app.run(debug = True)