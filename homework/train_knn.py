#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# descarga de datos
def prepare_data(file_path, test_size, random_state):

    df = pd.read_csv(file_path)

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test


def train_knn(x_train, y_train, n_neighbors=5):
    estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
    estimator.fit(x_train, y_train)
    return estimator


def print_metrics(estimator, x, y, label=""):
    y_pred = estimator.predict(x)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print()
    print(f"Metricas de {label}:")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")


# descarga de datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
x_train, x_test, y_train, y_test = prepare_data(url)

# dividir los datos en entrenamiento y testing

# entrenar el modelo
estimator = train_knn(x_train, y_train, n_neighbors=5)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
print_metrics(estimator, x_train, y_train, label="entrenamiento")

# Metricas de error durante testing
print_metrics(estimator, x_test, y_test, label="testing")
