import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QPushButton,
    QWidget, QInputDialog
)
from PyQt5.QtCore import Qt  # Добавлен импорт Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

            # Загрузка данных с указанным разделителем
            data = pd.read_csv(file_path, sep=sep)

            # Динамический выбор столбца с датами
            date_column, ok1 = QInputDialog.getItem(
                self, "Выбор столбца", "Выберите столбец с датами:", data.columns.tolist(), editable=False
            )
            if not ok1:
                raise ValueError("Выбор столбца с датами отменен.")

            # Динамический выбор целевого столбца
            target_column, ok2 = QInputDialog.getItem(
                self, "Выбор столбца", "Выберите целевой столбец для прогнозирования:", data.columns.tolist(), editable=False
            )
            if not ok2:
                raise ValueError("Выбор целевого столбца отменен.")

            # Преобразуем дату в формат datetime
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

            # Проверяем на наличие некорректных дат
            if data[date_column].isnull().any():
                raise ValueError("Некоторые значения в выбранном столбце дат некорректны.")

            # Сортируем данные по дате
            data = data.sort_values(by=date_column)

            # Подготовка данных
            X, y = self.prepare_data(data, target_column, date_column)

            # Генерация будущих дат для прогноза
            future_dates = self.generate_future_dates(data[date_column].max(), periods=2)

            # Построение прогноза
            model, forecast = self.build_forecast(X, y, future_dates)

            # Отображение графика
            self.plot_graph(data, target_column, date_column, forecast, future_dates)

        except Exception as e:
            self.label.setText(f"Ошибка: {e}")
            print(f"Ошибка: {e}")

    def prepare_data(self, data, target_column, date_column):
        try:
            # Преобразование столбца даты в числовой формат (например, год)
            data['date_numeric'] = data[date_column].dt.year

            # Убираем оригинальный столбец даты
            data = data.drop(columns=[date_column])

            # Разделяем данные на признаки (X) и целевую переменную (y)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            return X, y
        except Exception as e:
            raise ValueError(f"Ошибка при подготовке данных: {e}")

    def generate_future_dates(self, last_date, periods):
        try:
            # Генерация будущих дат с частотой 'Y' (годовой)
            return pd.date_range(start=last_date, periods=periods + 1, freq='Y')[1:]
        except Exception as e:
            raise ValueError(f"Ошибка при генерации будущих дат: {e}")

    def build_forecast(self, X, y, future_dates):
        try:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Обучение модели линейной регрессии
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Прогноз на тестовых данных
            predictions = model.predict(X_test)

            # Среднеквадратичная ошибка
            mse = mean_squared_error(y_test, predictions)
            print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

            # Прогноз на будущие даты
            future_X = pd.DataFrame({'date_numeric': future_dates.year})
            forecast = model.predict(future_X)

            return model, forecast
        except Exception as e:
            raise ValueError(f"Ошибка при построении прогноза: {e}")

    def plot_graph(self, data, target_column, date_column, forecast, future_dates):
        try:
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))

            # Реальные данные
            ax.plot(data[date_column], data[target_column], label="Реальные данные", color="blue")

            # Прогноз
            ax.plot(future_dates, forecast, label="Прогноз", color="red", linestyle="dashed")

            # Настройки графика
            ax.set_title("Прогнозирование данных")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Значение")
            ax.legend()

            # Отображение графика
            canvas = FigureCanvas(fig)
            self.setCentralWidget(canvas)
            canvas.draw()

        except Exception as e:
            raise ValueError(f"Ошибка при построении графика: {e}")


if __name__ == "__main__":

    variable_for_github = "someshittext"
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())