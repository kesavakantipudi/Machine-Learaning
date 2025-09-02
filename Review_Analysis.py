import pandas as pd
data = pd.read_csv('imdb.csv')

x = data['review']
y = data['sentiment']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


from sklearn.feature_extraction.text import CountVectorizer
vc = CountVectorizer(stop_words='english')
x_train = vc.fit_transform(x_train)
x_test = vc.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test)*100)
    
a = input("Enter the review: ")
a = vc.transform([a])
predict = nb.predict(a)

if predict[0] == 'positive':
    print('Positive')
else:
    print('Negative')