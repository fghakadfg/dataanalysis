#установка всех элементов интерфейса и модулей для взаимодействия с ними
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #настройка главного окна
        self.setWindowTitle("Анализ данных")
        self.setGeometry(100,100,100,100)

        #основной виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        #Создание интерфейса
        self.layout = QVBoxLayout()

        # Кнопка загрузки данных
        self.load_button = QPushButton("Загрузить данные")
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        # Метки для вывода информации
        self.info_label = QLabel("Здесь будет информация о данных")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label)

        self.central_widget.setLayout(self.layout)

    def load_data(self):
        # Диалог для выбора данных
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "CSV Files (*.csv);;All files (*)", options=options)
        if file_name:
            self.info_label.setText(f"Файл загружен: {file_name}")
