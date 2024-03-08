import pandas as pd
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
tweets = pd.read_csv('text.csv')
print(tweets.head(5))
print(len(tweets))

# data preprocessing
# Remove duplicate rows
tweets.drop_duplicates(inplace=True)
print(len(tweets))

# Remove rows with missing values
tweets.dropna(inplace=True)
print(len(tweets))

# Remove non-alphanumeric characters and convert to lowercase
tweets['text'] = tweets['text'].str.replace('[^a-zA-Z0-9]', ' ').str.lower()

# Remove stop words
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tweets['text'] = tweets['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(tweets.iloc[0]['text'])




# Print the number of unique tweets
print("Number of unique tweets: ", tweets['text'].nunique())

# Print the number of nulls in each column
print("Number of nulls in each column: ", tweets.isnull().sum())

# Generate a word cloud of the tweets
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(tweets['text']))
plt.figure(figsize=(15,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

print("we are here and we are going to print the most common words")

# Split the tweets into words, remove None values, and flatten the list
words = [word for sublist in tweets['text'].dropna().str.split().tolist() for word in sublist]

# Calculate the frequency distribution of the words
word_freq = nltk.FreqDist(words)
print("we are here")
# Print the top 10 most frequent words
print(word_freq.most_common(10))

# Calculate the length of each tweet
tweets['length'] = tweets['text'].str.len()

# Print the mean and standard deviation of the tweet lengths
print('Mean tweet length:', tweets['length'].mean())
print('Standard deviation of tweet length:', tweets['length'].std())

# Create a histogram of the tweet lengths
plt.hist(tweets['length'], bins=20)
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.title('Distribution of Tweet Lengths')
plt.show()


# feature extraction


# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the 'text' column
X = vectorizer.fit_transform(tweets['text'])

# Convert the result to a DataFrame
features = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Print the features
print(features)