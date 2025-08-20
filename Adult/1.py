import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据集
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# D:\\german\\Kernel-ridge-regression\\Adult\\adult\\adult.data
data = pd.read_csv("Adult\\adult\\adult.data", header=None, names=column_names)

# 2. 数据预处理
# 分离特征和目标变量
X = data.drop("income", axis=1)
y = data["income"]

# 确定分类特征和数值特征列
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建并训练模型
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. 保存模型（可选）
# from joblib import dump
# dump(model, "adult_income_model.joblib")


# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # 加载数据集
# column_names = [
#     "age", "workclass", "fnlwgt", "education", "education-num",
#     "marital-status", "occupation", "relationship", "race", "sex",
#     "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
# ]
# # 假设adult.data在当前目录下
# data = pd.read_csv("D:\\german\\Kernel-ridge-regression\\Adults\\adult\\adult.data", header=None, names=column_names)

# # 分离特征和目标变量
# X = data.drop("income", axis=1)
# y = data["income"]

# # 确定分类特征和数值特征列
# categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
# numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

# # 创建预处理管道
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numerical_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#     ]
# )

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 构建并训练逻辑回归模型
# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", LogisticRegression(max_iter=1000, random_state=42))
# ])
# model.fit(X_train, y_train)

# # 在测试集上进行预测
# y_pred = model.predict(X_test)

# # 评估模型
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # 也可以用训练好的模型对新数据进行预测（这里假设新数据是测试集的前5条）
# new_data = X_test.head()
# new_predictions = model.predict(new_data)
# print("Predictions for new data:\n", new_predictions)