import streamlit as slt
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Page settings
slt.set_page_config(
    page_title="Spam Mail Detection",
    page_icon="📧",
    layout="centered"
)


# Download NLTK data
@slt.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


download_nltk_data()

slt.title("Spam Mail Dection App")
slt.sidebar.title("📧 Spam Detection Info")

slt.sidebar.markdown("""
**How it works:**
- Uses machine learning to analyze email content
- Processes text with NLP techniques
- Provides instant spam/legitimate classification
""")
with slt.sidebar.expander("How to use"):
    slt.markdown("""
                1. **Paste or type** the email content in the text area above
                2. **Click the 'Check Email' button** to analyze the content
                3. **View the results** - the app will tell you if it's spam or legitimate
                4. **Check confidence level** to see how certain the model is about its prediction
                """)

# Download PDF option in sidebar
try:
    with open("manual.pdf", "rb") as pdf_file:
        pdf_data = pdf_file.read()
    slt.sidebar.download_button(
        label="📖 Download User Manual",
        data=pdf_data,
        file_name="spam_detection_manual.pdf",
        mime="application/pdf"
    )
except FileNotFoundError:
    slt.sidebar.error("Manual file not found")

slt.sidebar.markdown("""
**Features:**
- ✅ Real-time analysis
- ✅ High accuracy detection
- ✅ Simple one-click operation

**Instructions:**
1. Paste your email content
2. Click the 'Check' button
3. Get instant results

**Tips:**
- Include full email text for best results
- Works with any language email
- Safe and secure - no data stored

**Contact:**
- **Email:** hasiraza511@gmail.com
- **linkedin:** Muhammad Haseeb Raza AI Enginner
- **Phone:** +92193461511




""")


slt.sidebar.warning("⚠️ This is for educational purposes only")
slt.sidebar.info("💡 Tip: Longer emails give more accurate results")
main_model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf.pkl', 'rb'))
mail = slt.text_area("Enter your mail")
button = slt.button("Check")


def clean_text(text):
    # convert text into lower case
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-z\s]', '', text)  # [^a-zA-z\s] allow only these things allowed in text

    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_word = [word for word in words if word not in stop_words]

    # initialize porter stemmer
    stemmer = PorterStemmer()

    # Perform stemmer
    stemmed_words = [stemmer.stem(word) for word in filtered_word]
    # Join stemmed words
    cleaned_text = ''.join(stemmed_words)
    return cleaned_text


def predict(mail):
    cleand_mail= clean_text(mail)
    print(cleand_mail)
    result = main_model.predict(vectorizer.transform([cleand_mail]))
    if result[0] == 0:
        slt.error("fake mail")
    else:
        slt.success("Real mail")



if button:
    predict(mail)
else:
    slt.error("Please enter a valid mail")