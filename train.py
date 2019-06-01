import pandas as pd
from setup import get_train3, get_test3, get_valid3

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import sst
import math
from sklearn.linear_model import LogisticRegression, LinearRegression

train_data = get_train3()
test_data = get_test3()
val_data = get_valid3()

both = [train_data, test_data]
combined = pd.concat(both)

train_s = train_data.statement
test_s = test_data.statement
val_s = val_data.statement

#Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(combined)

X_train = vectorizer.transform(train_s)
X_test = vectorizer.transform(test_s)
X_val = vectorizer.transform(val_s)

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, train_data.label)

clf_6labels = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, train.label)


results_log_train = clf.predict(X_train)
results_log_test = clf.predict(X_test)

reg = LinearRegression().fit(X_train, train_data.simple_label)

results_lin_train = reg.predict(X_train)
results_lin_test = reg.predict(X_test)
print(results_lin_train)
print(results_lin_test)