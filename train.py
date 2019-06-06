from setup import get_train, get_test, get_valid
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys

'''
Args: num_labels (will include model choice later)
Usage e.x.: train.py 3
'''
num_labels = int(sys.argv[1])

train_data = get_train(num_labels)
test_data = get_test(num_labels)
val_data = get_valid(num_labels)

both = [train_data, test_data]
combined = pd.concat(both)

train_s = train_data.statement
test_s = test_data.statement
val_s = val_data.statement

#Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(combined.statement)

X_train = vectorizer.transform(train_s)
X_test = vectorizer.transform(test_s)
X_val = vectorizer.transform(val_s)


clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial', max_iter=500).fit(X_train, train_data.label)

print(clf.score(X_train, train_data.label))
print(clf.score(X_test, test_data.label))

reg = LinearRegression().fit(X_train, train_data.label)

results_lin_train = reg.predict(X_train)
results_lin_test = reg.predict(X_test)
print(results_lin_train.shape)

#print("6-Label Classification Train Score: ", clf_6labels.score(X_train, train_data.label))
#print("6-Label Classification Test Score: ", clf_6labels.score(X_test, test_data.label))

rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                             random_state=0)

rf.fit(X_train, train_data.label)

print(rf.feature_importances_)

print("Random Forest Classification Train Score: ", rf.score(X_train, train_data.label))
print("Random Forest Classification Test Score: ", rf.score(X_test, test_data.label))


et = ExtraTreesClassifier(n_estimators=100, max_depth=6,
                             random_state=0)

et.fit(X_train, train_data.label)

print("Extra Trees Classification Train Score: ", et.score(X_train, train_data.label))
print("Extra Trees Classification Test Score: ", et.score(X_test, test_data.label))

sgd = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, train_data.label)

print("SGD Classification Train Score: ", sgd.score(X_train, train_data.label))
print("SGD Classification Test Score: ", sgd.score(X_test, test_data.label))