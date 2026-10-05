from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "Congratulations! You won a free prize",
    "Win money now",
    "Hello, how are you?",
    "Can we meet tomorrow?",
    "Claim your free reward",
    "Please call me when you are free"
]

labels = ["spam", "spam", "ham", "ham", "spam", "ham"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

message = input("Enter a message: ")
message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)

print("Message:", message)
print("Prediction:", prediction[0])
