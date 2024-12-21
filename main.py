import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QPushButton,
    QWidget, QInputDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def add_time_features(data, date_column):
    """Добавляет временные признаки в данные."""
    data['year'] = data[date_column].dt.year
    data['month'] = data[date_column].dt.month
    data['day'] = data[date_column].dt.day
    data['day_of_week'] = data[date_column].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return data

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

            # Преобразуем дату в формат datetime
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

            # Проверяем на наличие некорректных дат
            if data[date_column].isnull().any():
                raise ValueError(f"Некоторые значения в столбце {date_column} некорректны.")

            # Динамический выбор целевого столбца
            target_column, ok2 = QInputDialog.getItem(
                self, "Выбор столбца", "Выберите целевой столбец для прогнозирования:", data.columns.tolist(), editable=False
            )
            if not ok2:
                raise ValueError("Выбор целевого столбца отменен.")

            # Проверяем, что целевой столбец не содержит некорректных значений
            data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            if data[target_column].isnull().any():
                raise ValueError(f"Некоторые значения в столбце {target_column} некорректны.")

            # Сортируем данные по дате
            data = data.sort_values(by=date_column)

            # Добавляем временные признаки
            data = add_time_features(data, date_column)

            # Определяем частоту данных (год или месяц)
            freq = pd.infer_freq(data[date_column])

            # Отбираем только необходимые столбцы для обучения
            selected_columns = ['year', 'month', 'day', 'day_of_week', 'is_weekend']
            X = data[selected_columns]  # Признаки
            y = data[target_column]  # Целевой столбец

            # Масштабируем данные
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Генерация будущих дат для прогноза
            future_dates = self.generate_future_dates(data[date_column].max(), periods=2, freq=freq)

            # Построение прогноза
            model, forecast = self.build_forecast(X_scaled, y, future_dates, scaler)

            # Отображение графика
            self.plot_graph(data, target_column, date_column, forecast, future_dates, freq)

        except Exception as e:
            self.label.setText(f"Ошибка: {e}")
            print(f"Ошибка: {e}")

    def generate_future_dates(self, last_date, periods, freq):
        """Генерация будущих дат в зависимости от частоты (по годам или по месяцам)."""
        try:
            # Генерация будущих дат с нужной частотой
            return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        except Exception as e:
            raise ValueError(f"Ошибка при генерации будущих дат: {e}")

    def build_forecast(self, X, y, future_dates, scaler):
        try:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Обучение модели градиентного бустинга
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            # Прогноз на тестовых данных
            predictions = model.predict(X_test)

            # Среднеквадратичная ошибка
            mse = mean_squared_error(y_test, predictions)
            print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")

            # Подготовка будущих данных для прогноза
            future_X = pd.DataFrame({
                'year': future_dates.year,
                'month': future_dates.month,
                'day': future_dates.day,
                'day_of_week': future_dates.dayofweek,
                'is_weekend': future_dates.dayofweek >= 5
            })

            future_X_scaled = scaler.transform(future_X)
            forecast = model.predict(future_X_scaled)

            return model, forecast
        except Exception as e:
            raise ValueError(f"Ошибка при построении прогноза: {e}")

    def plot_graph(self, data, target_column, date_column, forecast, future_dates, freq):
        try:
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))

            # Реальные данные
            ax.plot(data[date_column], data[target_column], label="Реальные данные", color="blue")

            # Прогноз
            ax.plot(future_dates, forecast, label="Прогноз", color="green", linestyle="dashed")

            # Настройки графика
            ax.set_title("Прогнозирование данных")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Значение")
            ax.legend()

            # Масштабирование оси X в зависимости от частоты
            if freq == 'A':  # Частота по годам
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            elif freq == 'M':  # Частота по месяцам
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))

            # Отображение графика
            canvas = FigureCanvas(fig)
            self.setCentralWidget(canvas)
            canvas.draw()

        except Exception as e:
            raise ValueError(f"Ошибка при построении графика: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    sys.exit(app.exec())
