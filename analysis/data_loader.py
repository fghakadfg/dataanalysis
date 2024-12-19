import pandas as pd

def load_csv(file_name):
    try:
        data = pd.read_csv(file_name, sep=";")
        return data
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        return None
