import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Set page title and background
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📩", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: red;
    }
    .header-text {
        color: #1f4773;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-text {
        color: orange;
        font-size: 32px;
        font-weight: bold;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and input area
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    st.subheader("Prediction Result:")
    if result == 1:
        st.markdown("<p class='result-text'>Spam 🚫</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-text'>Not Spam ✅</p>", unsafe_allow_html=True)
