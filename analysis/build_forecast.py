import pandas as pd  # pandas для работы с данными
from sklearn.linear_model import LinearRegression  # для построения модели линейной регрессии
import numpy as np  # для работы с массивами данных
import matplotlib.pyplot as plt  # для визуализации данных и прогноза


def build_forecast(data, target_column):
    try:
        if target_column not in data.columns:
            raise ValueError(f"Столбец '{target_column}' не найден в данных")

        X = data.drop(columns=[target_column])
        y = data[target_column]
        X = pd.get_dummies(X, drop_first=True)

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели линейной регрессии
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Прогноз для тестовых данных
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

        # Прогноз для будущих лет (например, 2021 и 2022)
        future_years = pd.DataFrame({'year': [2021, 2022]})  # Прогнозируем на 2021 и 2022 год
        future_X = pd.get_dummies(future_years, drop_first=True)  # Преобразуем в dummies (если нужно)

        forecast = model.predict(future_X)
        print(f"Прогноз для будущих лет: {forecast}")

        return model, forecast, data

    except Exception as e:
        print(f"Ошибка при построении прогноза: {e}")
        return None, None, None

# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    data = pd.read_csv("salary_data.csv")

    # Преобразуем столбец 'date' в числовые значения (например, извлечем год)
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year

    # Строим прогноз по зарплате для столбца 'salary'
    model, forecast = build_forecast(data, 'salary')
