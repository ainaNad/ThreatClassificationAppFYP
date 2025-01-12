import streamlit as st
import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import  matplotlib.pyplot as plt

df = pd.read_csv("annotated.csv")
#df = pd.read_csv('/Users/ainanadhirah/Downloads/annotated.csv')


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')


model = pickle.load(open("model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))





analyzer = SentimentIntensityAnalyzer()

def sentimentAnalysisVader (text) :
  sentiment_score = analyzer.polarity_scores(text)

  if sentiment_score['compound'] >= 0.05 :
        sentiment = 'Positive'
  elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
  else :
        sentiment = 'Neutral'

  negative_score = sentiment_score['neg']
  neutral_score  = sentiment_score['neu']
  positive_score = sentiment_score['pos']
  compound_score = sentiment_score['compound']

  return pd.Series([negative_score,neutral_score,positive_score,compound_score,sentiment])



def preprocess(text):
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower() #lowercase
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text) #url remove
    text = re.sub(r'[^\w\s]', '', text).replace('_', '').strip() #remove punct
    text = re.sub('\w*\d\w*','', text) #remove digit
    text = ' '.join(re.findall(r'[a-zA-Z\s]+', text)) #remove non english wors
    text = " ".join(text.split())
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    text = ' '.join(lemmatized_tokens)
    return text


def transformUserInput (text) :

    input_text = preprocess(text)
    transfomed_text = tfidf_vectorizer.transform([input_text])
    sentiment_inputetxt = sentimentAnalysisVader(input_text)
    compound_score = sentiment_inputetxt.iloc[3] 
    tfidf_density = transfomed_text.toarray()
    sentiment_feature = np.array([[compound_score]]) 
    combined_features = np.hstack([sentiment_feature,tfidf_density]) 
    prediction = model.predict(combined_features)
    
    print("predicted label :" , prediction)
    
    if prediction == 0 :
        return 'Unknown'
    elif prediction == 1 :
        return 'Irrelevant'
    elif prediction == 2 :
        return 'Business'
    elif prediction == 3 :
        return 'Threat'
    else :
        'unable to process output'
    


   
st.sidebar.title("Directory")
# page = st.sidebar.selectbox("Choose a page", ["Home", "About","Classify"])
page = st.sidebar.radio("Choose a page", ["Home", "About","Visualisation","Classify"])

if page == "Home":
    st.title("Welcome to the Cyber Threat Detector App")
    st.markdown("") 
    st.subheader("Home")
    st.write("This app allows you to input any text and returns the classification . It will classify the text into four main categories ")
    st.markdown("""
                - Irrelevant : Text may contain cyber threat terms but does not impose danger
                - Business : Text may contain marketing language
                - Threat : Text that imposes danger
                - Unknown : Text that falls in either of the category 

                """)
    st.subheader("Navigation")
    st.write("Hover over to the side bar to see more details on this app! ")
    st.write("1) Home - Main Page")
    st.write("2) About - Project Background")
    st.write("3) Visualisation - Top terms in each category")
    st.write("4) Classify - Classify Text Category")
    st.subheader("Important Notice")
    st.write("The classification is not 100% accurate .Thus , it is recommended to use with caution")

# About page
elif page == "About":
    st.title("About the App")

    st.subheader("Application background")
    st.write("""
    This project uses text data to detect emerging cyber threats.It combines machine learning and
    NLP techniques to predict the nature of a given text (e.g , business-related, irrelevant,threat,unknown).
    """)
    st.subheader("What is Cyber threat? ")
    st.write("""
    Cyber threat (also known as cybersecurity threat) refers to a malicious act that comprises a computers system, 
    network or data. It includes harmful activities such as data destruction, theft and disruption
    """)
    st.write("Some of the examples of cyber threats include:")
    st.markdown("""
                - Distributed Denial-of-Service (DDoS) Attack
                - Malware(e.g ,ransomeware,spyware)
                - SQL injection
                - Zero-day exploit

                """)
    st.subheader("Relationship between Social Media and Cyber Threats")
    st.write("""
    With the popularity of social media , cyber threats can happen at any moment with just one click.
    As social media is an readily available platform , it becomes easy for average social media users to exploit it for malicious purposes. 
    Many discussions and posts online revolves around cybersecurity topics be it about current news or experiences . 

    Thus, it is imperative to understand the category and emotional tone of these discussions as cyber threats continue 
    to proliferate. This issue must be addressed as cyber threats can interfere with one's life and cause social unrest.
    """)
    

# Visualization page
elif page == "Visualisation":
    st.title("Visualizations")
    st.write("""
    Here you can check out the top few terms in our current database for each category below :
    """)
    vectorizer = CountVectorizer(stop_words="english")
    df['preprocessed']= df['text'].apply(preprocess)
    X = vectorizer.fit_transform(df['preprocessed'])
    feature_names = vectorizer.get_feature_names_out()

    # Add the categories
    categories = df["annotation"].values

    # Map features to categories
    category_features = {}
    for category in np.unique(categories):
        category_indices = df[df["annotation"] == category].index
        category_matrix = X[category_indices, :]
        feature_scores = np.asarray(category_matrix.sum(axis=0)).flatten()
        top_features = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)
        category_features[category] = pd.DataFrame(top_features, columns=["Feature", "Frequency"])

    # Streamlit app

    for category, feature_df in category_features.items():
        st.subheader(f"Top terms for {category} Category")
        
        # Visualize in a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Feature", data=feature_df.head(10), color="blue", ax=ax)
        st.pyplot(fig)
        
# Predict page
elif page == "Classify":

 st.title("Threat Classification App")
 st.write("For each Sentence , this will return one of the 4 categories (Irrelevant, Business, Unknown , Threat)")
 st.markdown("""
                1) Enter text data 
                2) Submit by pressing on classify button
                3) Obtain classification
                """)
 st.markdown("") 
 x = st.text_input("Please Enter a Social media text post: ")


 if st.button('Classify'):
    if x:
        prediction = preprocess(x)
        st.write(f"Predicted Category: {transformUserInput(x)}")
    else :
        st.write("Please enter some text before pressing the button.")





