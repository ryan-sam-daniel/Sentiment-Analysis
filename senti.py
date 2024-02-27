import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
data=pd.read_csv('Valid.csv') 
x_train,x_test,y_train,y_test=train_test_split(data['text'],data['label'],test_size=0.2,random_state=42)
vectorizer=CountVectorizer()
x_train_vectors=vectorizer.fit_transform(x_train)
x_test_vectors=vectorizer.transform(x_test)
model=MultinomialNB()
model.fit(x_train_vectors,y_train)
y_pred=model.predict(x_test_vectors)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
test_valu=input('Enter the comments :' )
test_value=[test_valu]
test_value_vector=vectorizer.transform(test_value)
prediction=model.predict(test_value_vector)
for text,predictions in zip(test_value,prediction):
    print(f"text : {text}")
    print(f"predictions: {prediction}")
    print()