import re  # library for regular expression operations
import string  # for string operations
from collections import Counter

import nltk  # Python library for NLP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.corpus import twitter_samples
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt


df = pd.read_csv('Twitter Sentiments.csv')

print("DataFrame Information:")
print(df.info())
print('--------------------------------------------')
# Extract real usernames from the last column
real_usernames = df['username'].unique()


# Define a function to count the number of mentions in a tweet
def count_mentions(tweet):
    # Define a regular expression to match mentions in tweets
    mention_pattern = re.compile(r'@(\w+)')

    # Find all mentions in the tweet
    mentions = re.findall(mention_pattern, tweet)

    # Return the count of mentions
    return len(mentions)


# Output the number of unique users in the last column
print(f"Number of unique users in the last column: {len(real_usernames)}")

# Output the total number of mentions after replacing '@user' with real usernames
df['mention_count'] = df['tweet'].apply(count_mentions)

# Get the total number of mentions across all tweets
total_mentions = df['mention_count'].sum()

# Output the result
print(f"Total number of mentions in the dataset: {total_mentions}")
# remove twitter handles (@user)
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
print('--------------------------------------------')
print('remove twitter handles')
print(df.head())
# Display basic information about the DataFrame
print('--------------------------------------------')
# Display summary statistics of numeric columns
print("Summary Statistics of Numeric Columns:")
print(df.describe())

# Add a line break
print('--------------------------------------------')

# Display the first few rows of the DataFrame
print("First Few Rows of the DataFrame:")
print(df.head())
# remove special characters, numbers and punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
print('--------------------------------------------')
print('remove special characters, numbers and punctuations')
print(df.head())

# Count the number of tweets for each username (node)
tweets_count = df['username'].value_counts()

# Count the number of mentions for each username (node)
mention_counts = Counter()
for tweet in df['tweet']:
    mentions = [mention.strip('@') for mention in tweet.split() if mention.startswith('@')]
    mention_counts.update(mentions)

# Identify the top 15 nodes for each criterion
top_15_tweets_nodes = tweets_count.head(15)
top_15_mentions_nodes = [(username, count) for username, count in mention_counts.most_common() if username]
print('--------------------------------------------')

# Print the top 15 nodes and their corresponding counts for tweets
print("Top 15 Nodes for Number of Tweets:")
for i, (username, count) in enumerate(top_15_tweets_nodes.items(), start=1):
    print(f"{i}- Node: {username}, Tweets: {count}")

# Add a line break
print('--------------------------------------------')

mention_counts = Counter()
for tweet in df['tweet']:
    mentions = [mention.strip('@') for mention in tweet.split() if mention.startswith('@')]
    mention_counts.update(mentions)

# Identify the top 5 mentioned nodes excluding empty string
top_15_mentions_nodes = [(username, count) for username, count in mention_counts.most_common() if username]
# Print the top 5 mentioned nodes and their corresponding counts
print("Top 15 Nodes for Number of Mentions:")
for i, (username, count) in enumerate(top_15_mentions_nodes[:15], start=1):
    print(f"{i}- Node: {username}, Mentions: {count}")

# remove short words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
print('--------------------------------------------')
print('remove short words')
print(df.head())

# individual words considered as tokens
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
print('--------------------------------------------')
print('individual words considered as tokens')
print(tokenized_tweet.head())
# stem the words

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
print('--------------------------------------------')
print('stem words')
print(tokenized_tweet.head())

# combine words into single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

df['clean_tweet'] = tokenized_tweet
print('--------------------------------------------')
print('combine words into single sentence')
print(df.head())
print('--------------------------------------------')
# Download the "punkt" resource
nltk.download('punkt')


def get_keywords(tweet):
    # Tokenize the tweet
    tokens = word_tokenize(tweet.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    return filtered_tokens


df['keywords'] = df['tweet'].apply(get_keywords)

# Flatten the list of keywords and count their occurrences
all_keywords = [keyword for keywords in df['keywords'] for keyword in keywords]
keyword_counts = Counter(all_keywords)

# Get the 25 most common keywords
most_common_keywords = keyword_counts.most_common(25)
print('--------------------------------------------')
# Output the result
print("25 most used keywords in the dataset:")
for keyword, count in most_common_keywords:
    print(f"{keyword}: {count}")

# --------------------------------------------

# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# frequent words visualization for +ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# frequent words visualization for -ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

nltk.download('twitter_samples')
nltk.download('stopwords')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    # tweets: a list of tweets
    # ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)

    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    # freqs: a dictionary mapping each (word, sentiment) pair to its frequency

    return freqs


# create frequency dictionary
freqs = build_freqs(tweets, labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')

keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific', 'hate', 'never', 'fuck', 'disgust', 'unfair']
for keyword, count in most_common_keywords:
    keys.append(keyword)
print("keyskeyskeyskeyskeyskeyskeyskeyskeyskeys", keys)
# each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]
data = []

# loop through our selected words
for word in keys:

    # initialize positive and negative counts
    pos = 0
    neg = 0

    # retrieve number of positive counts
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]

    # retrieve number of negative counts
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]

    # append the word counts to the table
    data.append([word, pos, neg])
print('--------------------------------------------')
print('Word analysis: [word, positive frequency, negative frequency]')
print(data)
print('--------------------------------------------')
fig, ax = plt.subplots(figsize=(8, 8))

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data])

# Plot a dot for each pair of words
ax.scatter(x, y)

# assign axis labels
plt.xlabel("Log Positive count")
plt.ylabel("Log Negative count")

# Add the word as the label at the same position as you added the points just before
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0, 9], [0, 9], color='red')  # Plot the red line that divides the 2 areas.
plt.show()

freqs = build_freqs(train_x, train_y)
print('--------------------------------------------')

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))
print('--------------------------------------------')

print('This is an example of a positive tweet: \n', train_x[0])
print('--------------------------------------------')
print('This is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
print('--------------------------------------------')


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = -(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) / m

        # update the weights theta
        theta = theta - alpha * (np.dot(x.T, h - y)) / m

    J = np.mean(-y.T * np.log(h) - (1 - y).T * np.log(1 - h))

    return J, theta


def extract_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_l:

        if (word, 1.0) in freqs:
            # increment the word count for the positive label 1
            x[0, 1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            # increment the word count for the negative label 0
            x[0, 2] += freqs[(word, 0.0)]

    return x


# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

def neg(theta, pos):
    return (-theta[0] - pos * theta[1]) / theta[2]


fig, ax = plt.subplots(figsize=(10, 8))

colors = ['red', 'green']

# Color base on the sentiment Y
ax.scatter(X[:, 1].flatten(), X[:, 2].flatten(), c=[colors[int(k[0])] for k in Y], s=0.1)  # Flatten arrays if needed

plt.xlabel("Positive")
plt.ylabel("Negative")

# Now lets represent the logistic regression model in this chart.
maxpos = np.max(X[:, 1])  # max value in x-axis

# Plot a gray line that divides the 2 areas.
ax.plot([0, maxpos], [neg(theta, 0), neg(theta, maxpos)], color='gray')

plt.show()

# Extract usernames from tweet text
df['mentioned_users'] = df['tweet'].str.findall(r'@(\w+)')

# Create a dictionary to store the communities
communities = {}

# Iterate over each row in the dataframe
for _, row in df.iterrows():
    username = row['username']

    # Add the username to its own community if not already present
    if username not in communities:
        communities[username] = set()

    # Add mentioned users to the same community as the username
    for mentioned_user in row['mentioned_users']:
        communities[username].add(mentioned_user)

# Count the size of each community
community_sizes = {username: len(community) for username, community in communities.items()}

# Sort communities by size in descending order
sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)

# Display the top 5 communities
print('Top 10 communities with the most users:')
for i, (username, community_size) in enumerate(sorted_communities[:10], 1):
    print(f"{i}. Community for {username}: {', '.join(communities[username])} (Size: {community_size})")
