import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


class ForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Прогнозирование данных")
        self.setGeometry(100, 100, 1200, 800)

        # Переменные для хранения данных
        self.data = None
        self.target_column = None
        self.selected_features = None

        # Интерфейс
        self.initUI()

    def initUI(self):
        # Основной виджет и макет
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Загрузка файла
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Файл не выбран")
        self.file_button = QPushButton("Загрузить CSV файл")
        self.file_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        # Выбор целевого столбца
        target_layout = QHBoxLayout()
        self.target_label = QLabel("Целевой столбец не выбран")
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self.update_target_column)
        target_layout.addWidget(QLabel("Целевой столбец:"))
        target_layout.addWidget(self.target_combo)
        main_layout.addLayout(target_layout)

        # Кнопка для запуска анализа
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        main_layout.addWidget(self.run_button)

        # Графики
        self.canvas_forecast = FigureCanvas(plt.Figure(figsize=(5, 4)))
        self.canvas_importance = FigureCanvas(plt.Figure(figsize=(5, 4)))
        graph_layout = QHBoxLayout()
        graph_layout.addWidget(self.canvas_forecast)
        graph_layout.addWidget(self.canvas_importance)
        main_layout.addLayout(graph_layout)

        # Установка центрального виджета
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_file(self):
        try:
            # Открываем диалоговое окно для выбора файла
            file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл", "", "CSV Files (*.csv)")
            if not file_path:
                return

            # Загрузка данных
            sep, ok = QInputDialog.getText(
                self, "Разделитель CSV", "Введите разделитель (например, ',' или ';'):")
            if not ok or not sep:
                return

            self.data = pd.read_csv(file_path, sep=sep)
            self.file_label.setText(f"Файл: {file_path.split('/')[-1]}")

            # Заполнение целевого столбца
            self.target_combo.clear()
            self.target_combo.addItems(self.data.columns.tolist())
            self.run_button.setEnabled(True)

        except Exception as e:
            self.file_label.setText(f"Ошибка загрузки: {e}")

    def update_target_column(self):
        self.target_column = self.target_combo.currentText()

    def run_analysis(self):
        try:
            if self.data is None or self.target_column is None:
                raise ValueError("Не выбраны данные или целевой столбец.")

            # Подготовка данных
            X, y, column_transformer, feature_names = self.prepare_data()

            # Построение прогноза
            model, forecast, mse, next_index_forecast = self.build_forecast(X, y)
            print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

            # Отображение графиков
            self.plot_forecast(y, forecast, next_index_forecast)
            self.plot_feature_importances(model, feature_names)

        except Exception as e:
            print(f"Ошибка анализа: {e}")

    def prepare_data(self):
        selected_features = self.data.drop(columns=[self.target_column]).columns.tolist()
        X = self.data[selected_features]
        y = self.data[self.target_column]

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
        feature_names = numeric_features + list(column_transformer.named_transformers_['cat'].get_feature_names_out())

        return X_transformed, y, column_transformer, feature_names

    def build_forecast(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Прогноз на всем наборе данных
        forecast = model.predict(X)

        # Прогноз для следующего индекса
        next_features = np.asarray(np.mean(X, axis=0)).reshape(1, -1)
        next_index_forecast = model.predict(next_features)[0]

        return model, forecast, mse, next_index_forecast

    def plot_forecast(self, y, forecast, next_index_forecast):
        ax = self.canvas_forecast.figure.subplots()
        ax.clear()

        ax.plot(y.index, y, label="Реальные данные", color="blue")
        ax.plot(y.index, forecast, label="Прогноз", color="green", linestyle="dashed")
        ax.axvline(len(y), color="red", linestyle="dotted", label="Следующий прогноз")
        ax.scatter(len(y), next_index_forecast, color="red", label=f"Прогноз: {next_index_forecast:.2f}")

        ax.set_title("Прогнозирование данных")
        ax.set_xlabel("Индекс")
        ax.set_ylabel(self.target_column)
        ax.legend()
        self.canvas_forecast.draw()

    def plot_feature_importances(self, model, feature_names):
        ax = self.canvas_importance.figure.subplots()
        ax.clear()

        importances = model.feature_importances_
        ax.barh(feature_names, importances, color="skyblue")
        ax.set_xlabel("Важность признака")
        ax.set_title("Важность признаков")
        self.canvas_importance.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())
