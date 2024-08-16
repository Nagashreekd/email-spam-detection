import streamlit as streamlit
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Create a Porter Stemmer object
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('C:\Users\nagashree k d\Documents\machine learning project\vectorizer.pkl','rb'))
model = pickle.load(open('C:\Users\nagashree k d\Documents\machine learning project\model.pkl','rb'))

st,title("spam sms project")
input_sms = st.text_area("Enter message")

if st.button('predict'):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)
    if result == 1:
        st.header("spam")
        else:
            st.header("not spam")