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


st.title("Women's Job Search Portal")
user_name = st.text_input("Enter your name", "")
title = st.text_input("Enter your job title", "")
profile = st.text_input("Enter your preferred job profile", "")
user_mobile = st.text_input("Enter your contact details", "")

st.write("WELCOME TO EN-DRA ", user_name,". We hope to serve you with your best job match.")
Jobs = pd.read_csv('Job_Data.csv')
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

"""
#st.write(pd.DataFrame(title,Resumes))
#Resumes.shape()
#################################### SCORE CALCUATION ################################
@st.cache()
def calculate_scores(resumes, job_description):
    scores = []
    for x in range(resumes.shape[0]):
        score = Similar.match(
            resumes['TF_Based'][x], job_description['TF_Based'][index])
        scores.append(score)
    return scores


#Resumes['Scores'] = calculate_scores(Resumes, Jobs)

Ranked_resumes = Resumes.sort_values(
    by=['Scores'], ascending=False).reset_index(drop=True)

Ranked_resumes['Rank'] = pd.DataFrame(
    [i for i in range(1, len(Ranked_resumes['Scores'])+1)])



############################################ TF-IDF Code ###################################


@st.cache()
def get_list_of_words(document):
    Document = []

    for a in document:
        raw = a.split(" ")
        Document.append(raw)

    return Document


document = get_list_of_words(Resumes['Cleaned'])

id2word = corpora.Dictionary(document)
corpus = [id2word.doc2bow(text) for text in document]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                            update_every=3, chunksize=100, passes=50, alpha='auto', per_word_topics=True)

################################### LDA CODE ##############################################


@st.cache  # Trying to improve performance by reducing the rerun computations
def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.append(
                    [i, int(topic_num), round(prop_topic, 4)*100, topic_keywords])
            else:
                break

    return sent_topics_df
"""