import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
	return data[data.index == index]["title"].values[0]

def get_index_from_title(title):
	return data[data.title == title]["index"].values[0]
#read csv
data = pd.read_csv('movie_dataset.csv')

#select features
features = ['keywords','cast','genres','director']

for feature in features:
    data[feature]=data[feature].fillna('') #fill all the na 

#combine all features column

def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("Dc")
        # print(row)    


data["combined_features"]=data.apply(combine_features , axis=1)

# print(data['keywords'])
# print(data['cast'])
# print(data['genres'])
# print(data['director'])
print(data['combined_features'][0])

cv = CountVectorizer()
cnt_matrix = cv.fit_transform(data["combined_features"])
similarity  = cosine_similarity(cnt_matrix)

#get index of the movie from title
movie_user_likes = "Avatar"

movie_index = get_index_from_title(movie_user_likes)


similar_movie = list(enumerate(similarity[movie_index]))


sorted_similar_movie = sorted(similar_movie,key=lambda x:x[1],reverse=True)
# print(sorted_similar_movie)

#print the titles of the movie

i=0

for movie in sorted_similar_movie:
    
    print(get_title_from_index(movie[0]))
    i+=1
    if(i>50):
        break