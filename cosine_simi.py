from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]

vector = CountVectorizer()

matrix = vector.fit_transform(text)

# cnt_matrix = matrix.toarray()

# print(cnt_matrix)

similarity = cosine_similarity(matrix)

print(similarity)