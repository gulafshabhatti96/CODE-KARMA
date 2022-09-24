import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from operator import index
from wordcloud import WordCloud
from pandas._config.config import options
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import Similar
from PIL import Image
import time

from sklearn.feature_extraction.text import TfidfVectorizer


import nltk
import spacy
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Define english stopwords
stop_words = stopwords.words('english')

# load the spacy module and create a nlp object
# This need the spacy en module to be present on the system.
nlp = spacy.load('en_core_web_sm')
# proces to remove stopwords form a file, takes an optional_word list
# for the words that are not present in the stop words but the user wants them deleted.



st.markdown(f'<h1 style="color:#33ff33;font-size:65px;">{"EN-DRA"}</h1>', unsafe_allow_html=True)
Jobs = pd.read_csv('Job_Data.csv')


st.header("Women's Job Search Portal")
user_name = st.text_input("Enter your name", "")
title = st.text_input("Enter your job title", "")
profile = st.text_input("Enter your preferred job profile", "")
user_mobile = st.text_input("Enter your contact details", "")
st.write("WELCOME TO EN-DRA ", user_name,". We hope to serve you with your best job match.")
#title="school teacher"
user_mobile=7896451658

#profile="working of household school in college. Educator. Teacher"


data = [[title, profile, user_mobile]]
df = pd.DataFrame(data, columns=['title', 'profile','user_mobile'])

fig1 = go.Figure(data=[go.Table(
    header=dict(values=["job title", "job profile","contact detail"],
                fill_color='#00416d',
                align='center', font=dict(color='white', size=16)),
    cells=dict(values=[df.title, df.profile, df.user_mobile],
               fill_color='#d6e0f0',
               align='left'))])

fig1.update_layout(title="Your Searched Profile", width=700, height=1100)
st.write(fig1)

if st.button('Verify the details'): 
    
    def do_tfidf(token):
        tfidf = TfidfVectorizer() #max_df=0.05, min_df=0.002
        words = tfidf.fit_transform(token)
        sentence = " ".join(tfidf.get_feature_names())
        return sentence
    
    
    def remove_stopwords(text, stopwords=stop_words, optional_params=False, optional_words=[]):
        if optional_params:
            stopwords.append([a for a in optional_words])
        return [word for word in text if word not in stopwords]
    
    
    def tokenize(text):
        # Removes any useless punctuations from the text
        text = re.sub(r'[^\w\s]', '', text)
        return word_tokenize(text)
    
    
    def lemmatize(text):
        # the input to this function is a list
        str_text = nlp(" ".join(text))
        lemmatized_text = []
        for word in str_text:
            lemmatized_text.append(word.lemma_)
        return lemmatized_text
    
    # internal fuction, useless right now.
    
    
    def _to_string(List):
        # the input parameter must be a list
        string = " "
        return string.join(List)
    
    
    def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
        """
        Takes in Tags which are allowed by the user and then elimnates the rest of the words
        based on their Part of Speech (POS) Tags.
        """
        filtered = []
        str_text = nlp(" ".join(text))
        for token in str_text:
            if token.pos_ in postags:
                filtered.append(token.text)
        return filtered
    
    def _base_clean(text):
        """
        Takes in text read by the parser file and then does the text cleaning.
        """
        text = tokenize(text)
        text = remove_stopwords(text)
        text = remove_tags(text)
        text = lemmatize(text)
        return text
    
    
    def _reduce_redundancy(text):
        """
        Takes in text that has been cleaned by the _base_clean and uses set to reduce the repeating words
        giving only a single word that is needed.
        """
        return list(set(text))
    
    
    def _get_target_words(text):
        """
        Takes in text and uses Spacy Tags on it, to extract the relevant Noun, Proper Noun words that contain words related to tech and JD. 
    
        """
        target = []
        sent = " ".join(text)
        doc = nlp(sent)
        for token in doc:
            if token.tag_ in ['NN', 'NNP']:
                target.append(token.text)
        return target
    
    
    # https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
    # https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05
    
    def Cleaner(text):
        sentence = []
        sentence_cleaned = _base_clean(text)
        sentence.append(sentence_cleaned)
        sentence_reduced = _reduce_redundancy(sentence_cleaned)
        sentence.append(sentence_reduced)
        sentence_targetted = _get_target_words(sentence_reduced)
        sentence.append(sentence_targetted)
        return sentence
    
    def get_cleaned_words(document):
        for i in range(len(document)):
            raw = Cleaner(document[i])
            document.append(" ".join(raw[0]))
            document.append(" ".join(raw[1]))
            document.append(" ".join(raw[2]))
            sentence = do_tfidf(document[3].split(" "))
            document.append(sentence)
        return document
    @st.cache()
    def calculate_scores(resumes, job_description):
        scores = []
        for x in range(job_description.shape[0]):
            score = Similar.match(
                resumes['TF_Based'][0], job_description['TF_Based'][x])
            scores.append(score)
        return scores

    document=[]
    document.append(title+profile)
    
    Doc = get_cleaned_words(document)
    temp=[user_name]
    temp.extend(Doc)
    Database = pd.DataFrame([temp], columns=[
                            "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])
    
    Database.to_csv("women_Resume_Data.csv", index=False)
    Resumes = pd.read_csv('women_Resume_Data.csv')
    
    Jobs['Scores'] = calculate_scores(Resumes, Jobs)

    Ranked_jobs = Jobs.sort_values(        #Ranked_resumes--->Ranked_jobs
        by=['Scores'], ascending=False).reset_index(drop=True)
    
    Ranked_jobs['Rank'] = pd.DataFrame(
        [i for i in range(1, len(Ranked_jobs['Scores'])+1)])
    
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=["Job Name", "job profile"],
                    fill_color='#00416d',
                    align='center', font=dict(color='white', size=12)),
        cells=dict(values=[Ranked_jobs.Name[:5], Ranked_jobs.Context[:5]],
                   fill_color='#d6e0f0',
                   align='left'))])
    
    fig2.update_layout(title="Recommended Jobs", width=1200, height=500)
    st.write(fig2)




