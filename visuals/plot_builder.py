import matplotlib.pyplot as plt

def create_dynamic_bar_chart(data):
    """
    Строит столбчатую диаграмму для всех числовых столбцов.
    """
    numeric_columns = data.select_dtypes(include='number').columns
    non_numeric_columns = data.select_dtypes(exclude='number').columns

    if not numeric_columns.any():
        print("Нет числовых данных для построения графиков.")
        return

    x_labels = data[non_numeric_columns[0]] if non_numeric_columns.any() else data.index

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        plt.bar(x_labels, data[column], color='skyblue')
        plt.title(f"Значения для столбца: {column}")
        plt.xlabel(non_numeric_columns[0] if non_numeric_columns.any() else "Индексы")
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
