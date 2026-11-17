#importing libraries:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import nltk

#importing the dataset:

df = pd.read_csv("data/steam_reviews_processed.csv", encoding='latin1')

# #step no 1 (Data Cleaning):
# #checking null values and dropping them:


#Step2 (data filtering)

#finding the most mentioned game and its hour played:

most_mentioned_game = df['title'].value_counts().index[0]
most_mentioned_count =df['title'].value_counts().iloc[0]

print(f"the most mentioned game in the datset is: {most_mentioned_game} with {most_mentioned_count} reviews")

total_play_hours= df[df['title']==most_mentioned_game]
print(total_play_hours['hour_played'].describe())

#finding which game hold the max hours

max_hours= df['hour_played'].idxmax()

max_player_row = df.loc[max_hours]
print(f"Game with the most played hours: {max_player_row['title']}")
print(f"Hours played: {max_player_row['hour_played']}")

# #verall max hours grouped by game
max_by_game = df.groupby('title')['hour_played'].max().sort_values(ascending=False)
print(max_by_game.head(5))

#finding the decrease in playing hours and in which game
df['date_posted']= pd.to_datetime(df['date_posted'])

yearly_playtime= df.groupby(['title',df['date_posted'].dt.year])['hour_played'].mean().unstack()

yearly_playtime['drop'] = yearly_playtime.iloc[:,-2] - yearly_playtime.iloc[:,-1]

biggest_drop = yearly_playtime.sort_values(by='drop', ascending=False)
# print(biggest_drop['drop'].head()) 

# #evaluating the most extreme drop

extreme_drop_game = biggest_drop['drop'].index[0]

extreme_drop_value = biggest_drop['drop'].iloc[0]

print(f"The game with the highest play-hour drop is: {extreme_drop_game}")
print(f"Total average drop: {extreme_drop_value:.2f} hours")


exmaple = df['review'][50]
print(exmaple)

tokens= nltk.word_tokenize(exmaple)
print(tokens[:20])

pos= nltk.pos_tag(tokens)
print(pos[:10])

entities= nltk.chunk.ne_chunk(pos)
entities.pprint()

#VADER

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores(exmaple))

df['sentiment_score']= [
    sia.polarity_scores(str(text))['compound']
    for text in tqdm (df['review'].values, desc="Scoring Sentiment") 
]

df.to_csv('steam_reviews_processed.csv', index=False)

# MOST POSITIVE GAME WITH ITS HOUR PLAYED:

game_sentiment = df.groupby('title').agg ({
    'sentiment_score': 'mean',
    'hour_played': 'mean'
}).reset_index()

top_game = game_sentiment.sort_values(by='sentiment_score',ascending=False).iloc[0]

print(" Highest Positive Rated Game")
print(f"Game: {top_game['title']}")
print(f"Average Sentiment Score: {top_game['sentiment_score']:.4f}")
print(f"Average Hours Played: {top_game['hour_played']:.2f}\n")

# THE MOST NEGATIVE GAME WITH ITS REVIEW 

leat_review = df.groupby('title').agg({
    'sentiment_score' : 'mean',
    'hour_played' : 'mean'
}).reset_index()

bottom_game = game_sentiment.sort_values(by='sentiment_score', ascending= True).iloc[0]
worst_game = bottom_game['title']

print("Most Negative Rated Game")
print(f"Game: {worst_game}")
print(f"Average Sentiment Score: {bottom_game['sentiment_score']:.4f}")

negative_reviews = df[df['title'] == worst_game].sort_values(by='sentiment_score', ascending=True)['review'].dropna().head(3)
print("negative review snippet:")
for review in negative_reviews:
    print(f"--> {review[:500]}")
print("\n" + "="*40 + "\n")

# Most and Least Recommended Game

#converting recomendation colum from text to numeric values

# Convert recommendation column cleanly (handles True/False, booleans, and text strings)
df['rec_numeric'] = df['recommendation'].apply(lambda x: 1 if str(x).strip().lower() in ['true', '1', 'recommended', '1.0'] else 0)

# # Group by title and compute average recommendation rate
rec_summary = df.groupby('title')['rec_numeric'].mean().reset_index()

most_rec = rec_summary.sort_values(by='rec_numeric', ascending=False).iloc[0]
least_rec = rec_summary.sort_values(by='rec_numeric', ascending=True).iloc[0]

print(" Recommendation Breakdown ")
print(f"Most Recommended Game: {str(most_rec['title']).encode('ascii', 'ignore').decode()} ({most_rec['rec_numeric']*100:.1f}% positive)")
print(f"Least Recommended Game: {str(least_rec['title']).encode('ascii', 'ignore').decode()} ({least_rec['rec_numeric']*100:.1f}% positive)")
print("\n" + "="*40 + "\n")

# # Group by the early access column

print(" Early Access vs Full Release Analysis ")

early_access_col = 'is_early_access_review'

ea_summary = df.groupby(early_access_col)['sentiment_score'].agg(['mean', 'count']).reset_index()

for index, row in ea_summary.iterrows():
    # If True it's Early Access, if False it's Full Release
    status = "Early Access Review" if row[early_access_col] == True else "Full Release Review"
    print(f"Status: {status}")
    print(f"  - Average Sentiment Score: {row['mean']:.4f}")
    print(f"  - Total Review Count: {row['count']:,}")

print("\n" + "="*40 + "\n")

#recuuring negative terms for a game

from collections import Counter
import re
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

negative_texts = df[df['sentiment_score'] < -0.05]['review'].dropna()
print(f"Total negative reviews being analyzed: {len(negative_texts):,}")

stop_words = set(stopwords.words('english')).union(stopwords.words('spanish')).union({
    'game', 'games', 'play', 'played', 'playing', 'time', 'like', 'just', 
    'really', 'even', 'would', 'could', 'get', 'make', 'still', 'good', 'much',
    'your', 'there', 'will', 'when', 'more', 'this', 'that', 'from', 'with',
    'take', 'taking', 'every', 'want', 'dont', 'people', 'player', 'players' # filtering general pronouns/nouns to expose specific issues
})
words = []
for text in negative_texts:
    cleaned_words = re.findall(r'\b[a-zA-Z]{4,}\b', str(text).lower())
    words.extend([w for w in cleaned_words if w not in stop_words])

common_neg_word = Counter(words).most_common(10)

print("Top recurring words/complaints in negative reviews:")
for word, count in common_neg_word:
    print(f"  - '{word}': mentioned {count:,} times")


# --- Query 6: Playtime vs. Sentiment Analysis ---
print("\n--- Query 6: Playtime vs. Sentiment Correlation ---")

playtime_col = 'hour_played'

if playtime_col in df.columns:
    # 1. Pearson Correlation Coefficient
    correlation = df[playtime_col].corr(df['sentiment_score'])
    print(f"Overall Pearson Correlation between Playtime and Sentiment: {correlation:.4f}")
    
    # 2. Playtime Tiers
    bins = [-1, 10, 50, 100, 500, float('inf')]
    labels = ['< 10 hrs', '10-50 hrs', '50-100 hrs', '100-500 hrs', '500+ hrs']
    df['playtime_tier'] = pd.cut(df[playtime_col], bins=bins, labels=labels)
    
    print("\nAverage Sentiment Score by Playtime Tier:")
    tier_summary = df.groupby('playtime_tier', observed=False)['sentiment_score'].agg(['mean', 'count']).reset_index()
    for _, row in tier_summary.iterrows():
        print(f"  - {row['playtime_tier']:<12}: Avg Sentiment = {row['mean']:.4f} (Reviews: {row['count']:,})")
else:
    print(f"Playtime column '{playtime_col}' not found!")

print("\n" + "="*40 + "\n")