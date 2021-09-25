import pandas as pd
import numpy as np
credits = pd.read_csv("tmdb_5000_credits.csv")
movie_df = pd.read_csv("tmdb_5000_movies.csv")

print(credits.head())
print(movie_df.head())

print('Credits Shape::::',credits.shape)
print('Movies Dataframe:::::', movie_df.shape)

credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_df_merge = movie_df.merge(credits_column_renamed, on='id')
movies_df_merge.head()

movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned_df.head()

movies_cleaned_df.info()

#using weighted average for each movie's Average rating
""" W= (R * v + C * m)/(v+m)
W = Weighted Rating
R = average for the movie as a number from 0 to 10(mean) = (Rating)
v = number of votes for a movie = (votes)
m = minimum votes required to be listed in the Top 250(currently 3000)  --> If there is 5 stars rating but only one person has giver 5 stars than it is not appropriate.
so here we have selected 250 out of 3000 who had given 5 or higher stars.
C = the mean vote across the whole report (currently 6.9)"""

# Now create the variables which are given in formula
# calculate all the component based on the above formula.
v=movies_cleaned_df['vote_count']
R=movies_cleaned_df['vote_average']
C=movies_cleaned_df['vote_average'].mean()
m=movies_cleaned_df['vote_count'].quantile(0.70)   # which means that which value is more than 70%

movies_cleaned_df['weighted_average']=((R*v)+ (C*m))/(v+m)
print(movies_cleaned_df.head())


# Sort the average value from bigger to smaller
movie_sorted_ranking=movies_cleaned_df.sort_values('weighted_average',ascending=False)
print(movie_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20))
#print(movie_sorted_ranking.head())

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

weight_average=movie_sorted_ranking.sort_values('weighted_average',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10), data=weight_average)
plt.xlim(4, 10)
plt.title('Best Movies by average votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_movies.png')



popularity=movie_sorted_ranking.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,6))
ax=sns.barplot(x=popularity['popularity'].head(10), y=popularity['original_title'].head(10), data=popularity)

plt.title('Most Popular by Votes', weight='bold')
plt.xlabel('Score of Popularity', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_popular_movies.png')

from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()
movie_scaled_df = scaling.fit_transform(movies_cleaned_df[['weighted_average','popularity']])
movie_normalized_df = pd.DataFrame(movie_scaled_df,columns=['weighted_average','popularity'])
print('Normalized by weighted_average and popularity:::::', movie_normalized_df.head())


#create it into new movie datasets
movies_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = movie_normalized_df
print(movies_cleaned_df.head())

# here we are multiplying normalized_weight_average by 0.5 because we are giving 50% weight average to it. Similarly for popularity..
movies_cleaned_df['score'] = movies_cleaned_df['normalized_weight_average'] * 0.5 + movies_cleaned_df['normalized_popularity'] * 0.5
movies_scored_df = movies_cleaned_df.sort_values(['score'], ascending=False)
print(movies_scored_df[['original_title', 'normalized_weight_average', 'normalized_popularity', 'score']].head())


#plot that data
scored_df = movies_cleaned_df.sort_values('score', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')

#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
##
plt.savefig('scored_movies.png')

