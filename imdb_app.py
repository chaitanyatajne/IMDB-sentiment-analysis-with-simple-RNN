import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


nltk.download('wordnet')
leme = WordNetLemmatizer()
voc_size=3000


model=load_model('RNN.h5')

def preprocess(text):
    X=text
    corpus = []
    for i in range(0, len(X)):
        review = re.sub('[^a-zA-Z]', ' ', X[i])
        review = review.lower()
        review = review.split()

        review = [leme.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    onehot_repr=[one_hot(words,voc_size)for words in corpus]
    sent_length=40
    embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
    ar=np.array(embedded_docs)
    return ar

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')







