import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(url):
    df = pd.read_csv(url, sep=";")
    y = df["quality"]
    x = df.copy()
    x.pop("quality")
    return x, y


def split_data(x, y, test_size=0.25, random_state=123456):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_elasticnet(x_train, y_train, alpha=0.5, l1_ratio=0.5, random_state=12345):
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    estimator.fit(x_train, y_train)
    return estimator


def print_metrics(y_true, y_pred, stage=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print()
    print(f"Metricas de {stage}:")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")


# Main execution
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
x, y = load_data(url)
x_train, x_test, y_train, y_test = split_data(x, y)

estimator = train_elasticnet(x_train, y_train)
print()
print(estimator, ":", sep="")

# Metricas de entrenamiento
y_pred_train = estimator.predict(x_train)
print_metrics(y_train, y_pred_train, stage="entrenamiento")

# Metricas de testing
y_pred_test = estimator.predict(x_test)
print_metrics(y_test, y_pred_test, stage="testing")
