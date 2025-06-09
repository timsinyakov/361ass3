import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Read CSV
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# 1. 只在训练集上 fit
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['Text'])

# 2. 用同一个 vectorizer 转换测试集
X_test_tfidf  = tfidf_vectorizer.transform(df_test ['Text'])

encoder = {'entertainment':0, 'tech':1}
y_train_enc = df_train['Category'].map(encoder)
y_test_enc  = df_test ['Category'].map(encoder)

ks = list(range(1,50,2))
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    model.fit(X_test_tfidf, y_train_enc)
    y_predict = model.predict(X_test_tfidf)
    print("Test acc :", accuracy_score(y_test_enc, y_predict))
    print("Test F1  :", f1_score(y_test_enc, y_predict, average='weighted'))

# f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)


# # 交叉验证搜索
# gscv = GridSearchCV(knn, param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
# gscv.fit(X_train_tfidf, y_train_enc)

# # 输出最佳参数
# print("Best params:", gscv.best_params_)
# print("Best F1 score (CV):", gscv.best_score_)

# y_train = df_train['Category']
# y_test = df_test['Category']
# # 7) Tuning k (n_neighbors)
# k_range = range(1, 51, 2)
# scores = {}
# list_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
#     knn.fit(X_train_tfidf, y_train)
#     y_pred = knn.predict(df_train)
#     scores[k] = metrics.accuracy_score(y_test, y_pred)
#     list_scores.append(metrics.accuracy_score(y_test, y_pred))

# plt.plot(k_range, list_scores)
# plt.grid()
# plt.xlabel('Value of k')
# plt.ylabel('Accuracy')

