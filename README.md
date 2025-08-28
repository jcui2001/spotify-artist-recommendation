# Alternating Least Squares Recommendation Engine

### Types of Recommendation engines
1. Popularity of the songs
2. Content-based filtering
    - Get all the information about a song i.e. mood tags and genre tags, and match the taste based on the user's tastes
3. Collaborative-based filtering
    - Looks at tastes filtering
    - Disregards music tags and only looks at user's preferences
    - I.e. if two listener's are similar in their preference profile, they will also get recommended songs that one person liked
![image.png](attachment:6ce1db55-a14a-4a1a-9360-aac7224a82c2.png)

<img src="6ce1db55-a14a-4a1a-9360-aac7224a82c2.png" alt="image.png" width="300"/>

Spotify uses play logs and batch collaborative-filtering models to recommend songs.

## Matrix Factorization

![image.png](attachment:46e0cc76-6f0b-49c2-bb10-3a0cd6de5427.png)

Matrix factorization aims to reduce a large, sparse user item matrix into two matrices that are factors.

![image.png](attachment:7d643ee5-774c-4d1f-b537-22c52f7deeef.png)

### Alternating Least Squares

The user matrix has a row for each user and the artist matrix has a column for each artist.

Alternating least squares allows us to decide the number of factors we decompose the original user matrix into.

The larger the number of factors, the better we can re-construct the original user-artist matrix.

The advantage of collaborative filtering is that it allows us to avoid feature engineering because it creates the initial feature vectors on its own. The disadvantage of this is that, we don't actually know the features and embeddings.

### Background: Implicit Feedback Recommendation

Most recommendation algorithms (like classic collaborative filtering) assume explicit ratings (e.g., 1–5 stars).
But in practice, we often have implicit feedback:

Did the user click an item?

Did they listen to a song?

How many times did they view or purchase something?

We don’t have "negative feedback" — only positive signals and their strengths. The implicit library is designed for exactly this scenario.


### How do you find the value of a missing cell?

![image.png](attachment:9d1082f2-559c-4f43-8631-642f37d0f6fc.png)

Do the dot product of a particular row (1 user) in the user matrix by a particular column (1 artist) in the artist matrix.

## Recommendation Algorithm
1. Multiply a user vector with all artist vectors
    - This generates opinion values for all the artists for a given user
2. 
![image.png](attachment:36646f1a-5b66-4357-bdd6-cd17da991032.png)