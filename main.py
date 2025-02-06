import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QComboBox, QTableWidget, QTableWidgetItem, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ForecastWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, float, object, object, object, object)  # Передаем y_test и forecast

    def __init__(self, model, X_train, X_test, y_train, y_test, X_future, feature_names):
        super().__init__()
        self.model = model
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X_future = X_future
        self.feature_names = feature_names  # Добавляем feature_names

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        self.progress.emit(50)
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        forecast = self.model.predict(np.vstack([self.X_test, self.X_future]))
        self.progress.emit(100)
        self.finished.emit(self.model, forecast, mse, self.X_future, self.feature_names, self.y_test, forecast)  # Передаем y_test и forecast



class ForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогнозирование данных")
        self.setGeometry(100, 100, 1200, 800)
        self.data = None
        self.target_column = None
        self.y_test = None  # Добавим атрибут для хранения y_test
        self.forecast = None  # Добавим атрибут для хранения forecast
        self.models = {
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3,
                                                           random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression()
        }
        self.selected_model = "Gradient Boosting"
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.file_label = QLabel("Файл не выбран")
        self.file_button = QPushButton("Загрузить CSV")
        self.file_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        target_layout = QHBoxLayout()
        self.target_label = QLabel("Целевой столбец: ")
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self.update_target_column)
        target_layout.addWidget(self.target_label)
        target_layout.addWidget(self.target_combo)
        main_layout.addLayout(target_layout)

        model_layout = QHBoxLayout()
        self.model_label = QLabel("Выберите модель: ")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models.keys())
        self.model_combo.currentTextChanged.connect(self.update_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        self.forecast_steps_label = QLabel("Шаги прогнозирования: ")
        self.forecast_steps_combo = QComboBox()
        self.forecast_steps_combo.addItems([str(i) for i in range(1, 11)])
        main_layout.addWidget(self.forecast_steps_label)
        main_layout.addWidget(self.forecast_steps_combo)

        self.run_button = QPushButton("Запустить анализ")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["Факт", "Прогноз"])
        main_layout.addWidget(self.result_table)

        self.dark_mode_checkbox = QCheckBox("Тёмная тема")
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)
        main_layout.addWidget(self.dark_mode_checkbox)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.data = pd.read_csv(file_path)

            # Преобразуем столбец 'sales' в числовой формат, если это необходимо
            self.data['sales'] = pd.to_numeric(self.data['sales'], errors='coerce')

            # Преобразуем столбец 'date' в datetime
            self.data['date'] = pd.to_datetime(self.data['date'])

            self.file_label.setText(f"Файл: {file_path.split('/')[-1]}")
            self.target_combo.clear()
            self.target_combo.addItems(self.data.columns.tolist())
            self.run_button.setEnabled(True)

    def update_target_column(self):
        self.target_column = self.target_combo.currentText()

    def update_model(self, model_name):
        self.selected_model = model_name

    def run_analysis(self):
        if self.data is None or self.target_column is None:
            return

        selected_features = self.data.drop(columns=[self.target_column]).columns.tolist()
        X = self.data[selected_features]
        y = self.data[self.target_column]

        # Убедимся, что используем только числовые данные для расчета средних значений
        X_numeric = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        steps = int(self.forecast_steps_combo.currentText())

        # Создаем X_future, используя только числовые данные и их средние значения
        X_future = np.tile(X_numeric.mean().values, (steps, 1))

        self.progress_bar.setValue(10)
        model = self.models[self.selected_model]
        self.worker = ForecastWorker(model, X_train, X_test, y_train, y_test, X_future,
                                     X_numeric.columns)  # Передаем имена признаков
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.display_results)
        self.worker.start()

    def display_results(self, model, forecast, mse, X_future, feature_names, y_test, forecast_data):
        print(f"MSE: {mse:.2f}")
        self.y_test = y_test  # Сохраняем y_test
        self.forecast = forecast_data  # Сохраняем прогнозные данные

        # Используем все фактические данные (не только test выборку)
        y_all = pd.concat([self.data['sales'], pd.Series(forecast_data)], axis=0)  # Преобразуем forecast_data в Series
        x_all = np.arange(len(y_all))  # Индексы для всех данных (фактические и прогнозные)

        # Строим график
        self.figure1 = Figure(figsize=(8, 6))
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)

        # Строим график реальных данных (фактических)
        self.ax1.plot(self.data['date'], self.data['sales'], label="Реальные данные", color='blue', linestyle='-',
                      marker='o')  # Линия реальных данных

        # Строим прогноз (после реальных данных)
        forecast_dates = pd.date_range(self.data['date'].iloc[-1], periods=len(forecast),
                                       freq='D')  # Генерируем даты для прогнозных точек
        self.ax1.plot(forecast_dates, forecast, 'ro', label="Прогноз")  # Прогноз на будущее в виде точек

        # Подписи и легенда
        self.ax1.set_title("Прогнозирование")
        self.ax1.set_xlabel("Дата")  # Ось X - дата
        self.ax1.set_ylabel("Целевой показатель (sales)")  # Ось Y - целевой показатель
        self.ax1.legend()

        # Отображаем график
        self.create_new_window(self.canvas1)

        # График важности признаков, если модель поддерживает этот атрибут
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Строим график важности признаков
            self.figure2 = Figure(figsize=(8, 6))
            self.canvas2 = FigureCanvas(self.figure2)
            self.ax2 = self.figure2.add_subplot(111)

            # Строим столбчатую диаграмму
            self.ax2.barh(feature_names, importances, color='skyblue')

            # Подписи и легенда
            self.ax2.set_title("Важность признаков")
            self.ax2.set_xlabel("Важность")
            self.ax2.set_ylabel("Признаки")

            # Отображаем график
            self.create_new_window(self.canvas2)

    def create_new_window(self, canvas):
        # Создаем новое окно для отображения графика
        new_window = QMainWindow(self)
        new_window.setWindowTitle("График")
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        widget = QWidget()
        widget.setLayout(layout)
        new_window.setCentralWidget(widget)
        new_window.show()

    def toggle_dark_mode(self, state):
        if state:
            self.setStyleSheet("background-color: #333; color: white;")
        else:
            self.setStyleSheet("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())
