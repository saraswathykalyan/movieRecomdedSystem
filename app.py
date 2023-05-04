### Streamlit app

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title='Movie Recommendation System',page_icon="memo",layout="wide")
st.title('Movie Recommendation System')


st.sidebar.image('img.png')

def get_recommendation(movie):
    df = pd.read_csv('movie_dataset.csv')

    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    ratings.sort_values('num of ratings',ascending=False).head(10)

    moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

    movie_user_ratings = moviemat[movie]
    similar_movies = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_movies,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    corr_movie = corr_movie.join(ratings['rating'])
    
    recommended_movies = corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    recommended_movies_df = recommended_movies.iloc[1:11,:]
    return recommended_movies_df

df = pd.read_csv('movie_dataset.csv')
movies_list = list(df['title'].unique())
movie_title = st.sidebar.selectbox('Select a movie:', movies_list)

if st.sidebar.button('Get Recommendations'):
    st.subheader('Top 10 Similar Movies:')
    top_n_movies = get_recommendation(movie_title)
    
    recommended_movies  = list(top_n_movies.index)
    recommended_movies_df = pd.DataFrame(recommended_movies,columns=['Recommended Movie'])
    
    st.table(recommended_movies_df)
    
    # plot top 10 similar movies
    top_n_movies['similarity_score'] = pd.to_numeric(top_n_movies['Correlation'])
    fig1, ax1 = plt.subplots()
    sns.barplot(x='similarity_score', y=top_n_movies.index, data=top_n_movies, palette='Blues_d')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Movie Title')
    ax1.set_title('Top 10 Similar Movies')
    st.pyplot(fig1)
    
    # plot top 10 recommended movies for user
    top_n_movies['predicted_rating'] = pd.to_numeric(top_n_movies['num of ratings'])
    fig2, ax2 = plt.subplots()
    sns.barplot(x='predicted_rating', y=top_n_movies.index, data=top_n_movies, palette='Oranges_d')
    ax2.set_xlabel('Predicted Rating')
    ax2.set_ylabel('Movie Title')
    ax2.set_title('Top 10 Recommended Movies for User')
    st.pyplot(fig2)
    
    # Plot a histogram of the ratings distribution for the recommended movies
    fig, ax = plt.subplots()
    sns.histplot(top_n_movies['rating'], bins=10, ax=ax)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ratings for Recommended Movies')
    st.pyplot(fig)
    
    # Plot a bar chart of the number of ratings for the recommended movies
    fig, ax = plt.subplots()
    chart_data = top_n_movies.groupby('title')['rating'].sum().reset_index()
    chart_data.columns = ['title', 'num_ratings']
    chart_data = chart_data.sort_values(by='num_ratings', ascending=False).head(10)
    sns.barplot(x='num_ratings', y='title', data=chart_data, ax=ax)
    ax.set_xlabel('Number of Ratings')
    ax.set_ylabel('Movie Title')
    ax.set_title('Number of Ratings for Recommended Movies')
    st.pyplot(fig)