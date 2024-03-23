import pandas as pd
import numpy as np
import random
from wordcloud import WordCloud
from matplotlib import pyplot as plt
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

# Example usage:
documents = [["word1", "word2"], ["word2", "word3"], ["word1", "word3"]]
print(mstream(documents, k=3, i=2))

import pandas as pd

# Read the CSV file
df = pd.read_csv('text.csv', header=None)

# Convert the text column to a list of words
documents = df[1].str.split().tolist()

# Run the mstream function
clusters = mstream(documents, k=3, i=2)

# Print the result
print(clusters)

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
        print(f'Cluster {cluster_idx} does not contain any words')



# Assume that `true_labels` is a list of the true labels for each document
true_labels = df[2].tolist()  # Replace with your code to get the true labels

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

# Get the features (words) from the vectorizer
features = vectorizer.get_feature_names_out()

# For each cluster
for i in range(3):
    # Get the centroid of the cluster
    centroid = kmeans.cluster_centers_[i]
    
    # Sort the features by their importance in the centroid
    sorted_features = sorted(zip(features, centroid), key=lambda x: x[1], reverse=True)
    
    # Get the top 10 features
    top_features = dict(sorted_features[:10])
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_features)
    
    # Display the word cloud
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Print the metrics for the KMeans algorithm
print(f"KMeans ARI: {kmeans_ari}, NMI: {kmeans_nmi}, FMI: {kmeans_fmi}")

# # Compute the metrics for the mstream algorithm
# mstream_ari = metrics.adjusted_rand_score(true_labels, predicted_labels)
# mstream_nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)
# mstream_fmi = metrics.fowlkes_mallows_score(true_labels, predicted_labels)

# Print the metrics for the mstream algorithm
print(f"MStream ARI: {mstream_ari}, NMI: {mstream_nmi}, FMI: {mstream_fmi}")


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