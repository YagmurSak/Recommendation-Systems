import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Data Preparation

movie = pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\4-Recommendation Systems\HybridRecommender-221114-235254\datasets\movie.csv")
rating = pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\4-Recommendation Systems\HybridRecommender-221114-235254\datasets\rating.csv")
df = movie.merge(rating, how="left", on="movieId")
df.head()

# Data Analysis and First Insights

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Calculating the total number of votes for each movie

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()

# Remove the movies with a total number of votes below 1000 from the dataset.

rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

common_movies = common_movies.reset_index()
common_movies.head()

# We create a pivot table that shows the ratings each user gives to the movies

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.head()

# Determining the movies watched by the user to be suggested and creating new dataframe

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()


# Accessing Data and IDs of Other Users Watching the Same Movies

user_movie_df.loc[user_movie_df.index == random_user]

movies_watched_df = user_movie_df[movies_watched]

# Create a new dataframe named user_movie_count that contains information about how many of the movies the selected
# user has watched for each user.

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count.head()

# We create a list called users_same_movies from the IDs of users who watched 60 percent and above of the
# movies voted by the selected user.

perc = len(movies_watched) * 60 / 100

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

final_df.columns = pd.Index([f"{col}_{i}" if final_df.columns.duplicated()[i] else col for i, col in enumerate(final_df.columns)])

final_df.head()

final_df.shape

# We create a new corr_df dataframe where the correlations between users will be found.

corr_df = final_df.T.corr().unstack().sort_values()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]

# Create a new dataframe by filtering out users that have a high correlation (above 0.65) with the selected user

top_users = top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Merge the top_users dataframe with the rating dataset

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

top_users_ratings.head()

# Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

top_users_ratings.head()

# Create a new dataframe containing the movie id and the average value of all users' weighted ratings for each movie.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df.head()

# Select the movies with a weighted rating greater than 3.5 in recommendation_df and sort them by weighted rating.
# The first 5 observations will be movies to be recommend.

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

#### Item-Based Recommendation ####
#
# We will make item-based recommendations based on the name of the movie the user last watched and gave the highest rating to.

user = 108170

# We get the ID of the movie with the most up-to-date score from the movies that the user gave 5 points to recommend.

film_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
film_id

film_df = user_movie_df[movie[movie["movieId"] == film_id]["title"].values[0]]
film_df.head()

user_movie_df.corrwith(film_df).sort_values(ascending=False).head(10)

