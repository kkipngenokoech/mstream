# -*- coding: utf-8 -*-
"""olap.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xF69lF1xBzwq1I2aCcUQrWpRhSgDGoQ6

**DATA LOADING**

here we load our dataset from our google drive and print out some pages just to ensure we have actually loaded the datasets
"""

import pandas as pd
tweets = pd.read_csv('/content/drive/MyDrive/text.csv')
print(tweets.head())
print(len(tweets))

"""**DATA PREPOCESSING & Exploratory Data Analysis (EDA)**

we are cleaning our data here by:
1. Removing duplicate rows
2. Removing rows with missing values
3. Removing non-alphanumeric characters and coverting to lowercase
4. Removing stop words

After cleaning data we perform an Exploratoy Data anaylsis on the data which includes:
1. word cloud - to visualize the most frequent words in a corpus of text data
2. word frequency analysis - to anaylse the most common words in the corpus
3. Tweet Length Analysis - this depicts the distribution of tweet lengths, showing how frequently tweets occur at different length intervals.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Remove duplicate rows
tweets.drop_duplicates(inplace=True)

# Remove rows with missing values
tweets.dropna(inplace=True)

# Remove non-alphanumeric characters and convert to lowercase
tweets['text'] = tweets['text'].str.replace('[^a-zA-Z0-9]', ' ').str.lower()

# Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
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



# Split the tweets into words, remove None values, and flatten the list
words = [word for sublist in tweets['text'].dropna().str.split().tolist() for word in sublist]

# Calculate the frequency distribution of the words
word_freq = nltk.FreqDist(words)

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

"""**MSTREAM ALGORITHM**

Given a list of documents (tweets) as input, along with optional parameters k and i, the function initializes k+1 empty clusters and iterates through each document in a one-pass manner.

For each document, it calculates probabilities for assigning the document to existing clusters or creating a new cluster. By sampling a cluster index based on these probabilities, the function updates the clustering accordingly, either by assigning the document to an existing cluster or creating a new cluster if necessary.

At the end of the process, the function returns the clusters formed during this one-pass iteration. This approach enables efficient clustering of streaming data while maintaining low memory usage and computational complexity.
"""

import random
from sklearn import metrics
def mstream(documents, k=3, i=2):
    # Initialization
    clusters = {idx: {} for idx in range(k+1)}
    mz = 0
    nz = 0

    # One pass clustering process
    for doc_idx, doc in enumerate(documents):
        # Compute probabilities for existing clusters and a new cluster
        probabilities = []
        for cluster_idx in range(k+1):
            if cluster_idx == k:
                probabilities.append(1 / (k+1))
            else:
                probabilities.append(sum([clusters[cluster_idx].get(w, 0) for w in doc]))

        # Sample cluster index for document
        sampled_probability = random.random()
        cumulative_probability = 0
        for i, p in enumerate(probabilities):
            cumulative_probability += p
            if cumulative_probability >= sampled_probability:
                cluster_idx = i
                break
        else:
            cluster_idx = k

        # Update clustering process
        if cluster_idx == k:
            k += 1
            clusters[k] = {}
            mz += 1
            nz += len(doc)

        for word in doc:
            clusters[cluster_idx][word] = clusters[cluster_idx].get(word, 0) + 1

    return clusters
print(tweets.head())
documents = tweets["text"].str.split().tolist()

# Run the mstream function
clusters = mstream(documents, k=3, i=2)

for cluster_idx, cluster in clusters.items():
    # Convert the cluster to a string of words
    cluster_string = ' '.join([' '.join([word]*count) for word, count in cluster.items()])

    # Check if the cluster contains at least one word
    if cluster_string:
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400).generate(cluster_string)

        # Display the word cloud
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {cluster_idx}')
        plt.show()
    else:
      pass
        #print(f'Cluster {cluster_idx} does not contain any words')



# Assume that `true_labels` is a list of the true labels for each document
true_labels = tweets["label"].tolist()  # Replace with your code to get the true labels

# Assume that `predicted_labels` is a list of the predicted labels for each document
# Convert clusters to predicted labels
predicted_labels = []
for doc in documents:
    for cluster_idx, cluster in clusters.items():
        if any(word in cluster for word in doc):
            predicted_labels.append(cluster_idx)
            break
    else:
        predicted_labels.append(-1)  # Label for documents not assigned to any cluster

# Compute ARI, NMI, and FMI
mstream_ari = metrics.adjusted_rand_score(true_labels, predicted_labels)
mstream_nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)
mstream_fmi = metrics.fowlkes_mallows_score(true_labels, predicted_labels)

print(mstream_ari)

"""**KMEANS algorithm**

We apply KMeans clustering algorithm to our cleaned tweets, aiming to partition them into distinct groups based on their content.

Initially, the tweets undergo a transformation into a TF-IDF feature matrix, where each tweet is represented as a vector capturing the significance of words within it relative to the entire tweet collection.

KMeans clustering is then performed on this feature matrix, with the number of clusters set to three.

Subsequently, we compute three evaluation metrics:
1. Adjusted Rand Index (ARI)
2. Normalized Mutual Information (NMI)
3. Fowlkes-Mallows Index (FMI)

to assess the agreement between the true labels of the tweets and the clusters assigned by KMeans.

Finally, we print the metrics to provide insights into the effectiveness of KMeans in uncovering meaningful clusters within the tweet dataset.
"""

# KMEANS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

# Convert the documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([' '.join(doc) for doc in documents])

# Run the KMeans algorithm
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Get the predicted labels
kmeans_predicted_labels = kmeans.labels_

# Compute the metrics for the KMeans algorithm
kmeans_ari = metrics.adjusted_rand_score(true_labels, kmeans_predicted_labels)
kmeans_nmi = metrics.normalized_mutual_info_score(true_labels, kmeans_predicted_labels)
kmeans_fmi = metrics.fowlkes_mallows_score(true_labels, kmeans_predicted_labels)

# Print the metrics for the KMeans algorithm
print(f"KMeans ARI: {kmeans_ari}, NMI: {kmeans_nmi}, FMI: {kmeans_fmi}")

"""**comparing the two algorithms**

Here we are comparing our Mstream Algorithm and Kmeans algorithm basing on the three metrics
"""

from sklearn import metrics
import numpy as np
# Metrics for the KMeans algorithm
kmeans_metrics = [kmeans_ari, kmeans_nmi, kmeans_fmi]
print(f'keans {kmeans_metrics}')
# Metrics for the mstream algorithm
mstream_metrics = [mstream_ari, mstream_nmi, mstream_fmi]
print(f'mstream {mstream_metrics}')
# The label locations
x = np.arange(len(kmeans_metrics))

# The width of the bars
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, kmeans_metrics, width, label='KMeans')
rects2 = ax.bar(x + width/2, mstream_metrics, width, label='MStream')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores (log scale)')
ax.set_title('Scores by algorithm and metric')
ax.set_xticks(x)
ax.set_xticklabels(['ARI', 'NMI', 'FMI'])
ax.legend()

# Use a logarithmic scale for the y-axis
ax.set_yscale('log')

fig.tight_layout()

plt.show()

"""**CONCLUSION**

**Adjusted Rand Index (ARI)**: The ARI for KMeans is significantly higher than for MStream. This suggests that KMeans is doing a better job at matching the ground truth clusters in our data.

**Normalized Mutual Information (NMI)**: Similar to ARI, the NMI for KMeans is also significantly higher than for MStream. This again suggests that KMeans is doing a better job at matching the ground truth clusters.

**Fowlkes-Mallows Index (FMI)**: The FMI for MStream is higher than for KMeans, which suggests that MStream might be doing a better job in terms of the geometric mean of precision and recall.


"""