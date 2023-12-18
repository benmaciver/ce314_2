import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#helper functions
def remove_stop_words_from_2d_array(arr):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for row in arr:
        filtered_sentence.append([])
        for w in row:
            if w not in stop_words:
                filtered_sentence[-1].append(w)
    return filtered_sentence
def remove_non_alphabetical_words_from_2d_array(arr):
    filtered_sentence = []
    for row in arr:
        filtered_sentence.append([])
        for w in row:
            if w.isalpha():
                filtered_sentence[-1].append(w)
    return filtered_sentence

def split_2d_arr(arr):
    for x in range(len(arr)):
        lastElement = arr[x][1]
        review = []
        review = arr[x][0].split()
        review.append(lastElement)
        arr[x] = review
    return arr

def filter_2d_arr(arr):
    arr = split_2d_arr(arr)
    arr = remove_stop_words_from_2d_array(arr)
    arr = remove_non_alphabetical_words_from_2d_array(arr)
    return arr



# Open the CSV file for reading
with open('imdb.csv', 'r', encoding='utf-8') as csvfile:
    # Create a csvreader object
    reader = csv.reader(csvfile, delimiter=',')
    #skip first line  
    next(reader)
    # Read the data into a 2D array
    data = []
    for row in reader:
        data.append(row)
        

#get data        
training_data = data[:40000]
test_data = data[-10000:]


#filter data
training_data = filter_2d_arr(training_data)
test_data = filter_2d_arr(test_data)



# Flatten the 2D array to a list of words
flat_training_data = [word for row in training_data for word in row]

# Calculate word frequencies
word_freq = Counter(flat_training_data)
# Report linguistic features
print("Total words in training data:", len(flat_training_data))
print("Unique words in training data:", len(word_freq))
print("Top 10 most common words:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq} times")


#removes the positive or negative tag from the test data
x_test = []
y_test = []
for x in range (len(test_data)):
    x_test.append(test_data[x][:])
    y_test.append(test_data[x][-1])

x_train = []
y_train = []
for x in range (len(training_data)):
    x_train.append(training_data[x][:])
    y_train.append(training_data[x][-1])

x_train_tfidf = TfidfVectorizer().fit_transform(x_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)











