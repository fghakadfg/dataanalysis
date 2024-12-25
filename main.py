import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QPushButton,
    QWidget, QInputDialog
)
from PyQt5.QtCore import Qt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


class ForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Прогнозирование данных")
        self.setGeometry(100, 100, 800, 600)

        # Основной интерфейс
        self.label = QLabel("Загрузите CSV файл для анализа", self)
        self.label.setStyleSheet("font-size: 18px;")
        self.label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Загрузить CSV файл", self)
        self.button.clicked.connect(self.load_file)

        # Настройка центрального виджета
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_file(self):
        try:
            # Открываем диалоговое окно для выбора файла
            file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл", "", "CSV Files (*.csv);;All Files (*)")
            if not file_path:
                self.label.setText("Файл не выбран.")
                return

            # Запрашиваем разделитель у пользователя
            sep, ok = QInputDialog.getText(
                self, "Разделитель CSV", "Введите разделитель (например, ',' для запятых или ';' для точек с запятой):"
            )
            if not ok or not sep:
                raise ValueError("Разделитель не указан.")

            # Загрузка данных
            data = pd.read_csv(file_path, sep=sep)

            # Выбор целевого столбца
            target_column, ok1 = QInputDialog.getItem(
                self, "Выбор столбца", "Выберите целевой столбец для прогнозирования:", data.columns.tolist(), editable=False
            )
            if not ok1:
                raise ValueError("Выбор целевого столбца отменен.")

            # Предложение выбрать временной столбец (необязательно)
            date_column, ok2 = QInputDialog.getItem(
                self, "Выбор столбца", "Выберите столбец с датами (или оставьте пустым):", [''] + data.columns.tolist(), editable=True
            )
            if ok2 and date_column:
                data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
                if data[date_column].isnull().any():
                    raise ValueError("Некоторые значения в выбранном столбце дат некорректны.")
                data = self.add_time_features(data, date_column)

            # Выбор признаков
            features, ok3 = QInputDialog.getMultiLineText(
                self, "Выбор признаков",
                "Введите столбцы через запятую, которые будут использоваться как признаки (оставьте пустым для всех):"
            )
            if ok3 and features.strip():
                selected_features = [col.strip() for col in features.split(",")]
            else:
                selected_features = data.drop(columns=[target_column]).columns.tolist()

            # Подготовка данных
            X, y, column_transformer, feature_names = self.prepare_data(data, selected_features, target_column)

            # Построение прогноза
            model, forecast, mse, next_index_forecast = self.build_forecast(X, y)
            print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

            # Отображение графиков
            self.plot_graph(data, forecast, y, target_column, date_column, next_index_forecast)
            self.plot_feature_importances(model, feature_names)

        except Exception as e:
            self.label.setText(f"Ошибка: {e}")
            print(f"Ошибка: {e}")

    def add_time_features(self, data, date_column):
        """Добавляет временные признаки в данные."""
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        data['day'] = data[date_column].dt.day
        data['day_of_week'] = data[date_column].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        return data

    def prepare_data(self, data, selected_features, target_column):
        """Подготавливает данные для анализа."""
        X = data[selected_features]
        y = data[target_column]

        # Разделяем признаки на числовые и категориальные
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Преобразование данных
        column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        X_transformed = column_transformer.fit_transform(X)

        # Восстановление имен признаков
        feature_names = list(column_transformer.named_transformers_['num'].get_feature_names_out(numeric_features)) + \
                        list(column_transformer.named_transformers_['cat'].get_feature_names_out())

        return X_transformed, y, column_transformer, feature_names

    def build_forecast(self, X, y):
        """Строит модель прогнозирования и возвращает прогнозы."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Прогноз на всем наборе данных
        forecast = model.predict(X)

        # Прогноз для следующего индекса
        next_index_forecast = self.predict_next(model, X)

        return model, forecast, mse, next_index_forecast

    def predict_next(self, model, X):
        """Строит прогноз для следующего индекса."""
        # Среднее значение по признакам для предположительного прогноза
        next_features = np.mean(X, axis=0).reshape(1, -1)
        next_prediction = model.predict(next_features)
        return next_prediction[0]

    def plot_graph(self, data, forecast, y, target_column, date_column, next_index_forecast):
        """Отображает график прогноза и выводит значения прогнозов и реальных данных в консоль."""
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, y, label="Реальные данные", color="blue")
        plt.plot(data.index, forecast, label="Прогноз", color="green", linestyle="dashed")

        # Добавление прогноза для следующего индекса
        plt.axvline(len(data.index), color="red", linestyle="dotted", label="Следующий прогноз")
        plt.scatter(len(data.index), next_index_forecast, color="red", label=f"Прогноз: {next_index_forecast:.2f}")

        plt.title("Прогнозирование данных")
        plt.xlabel("Индекс")
        plt.ylabel(target_column)
        plt.legend()
        plt.show()

        # Вывод значений прогнозов и реальных данных в консоль
        print("\nСравнение прогнозов и реальных данных:")
        for idx, (real, pred) in enumerate(zip(y, forecast)):
            print(f"Индекс: {idx}, Реальное значение: {real}, Прогноз: {pred:.2f}")

        # Вывод прогноза для следующего индекса
        print(f"\nПрогноз для следующего индекса ({len(data.index)}): {next_index_forecast:.2f}")

    def plot_feature_importances(self, model, feature_names):
        """Отображает график важности признаков."""
        # Важность признаков
        importances = model.feature_importances_

        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color="skyblue")
        plt.xlabel("Важность признака")
        plt.title("Важность признаков")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())
