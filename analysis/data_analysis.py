def analyze_data(data):
    """
    Возвращает статистическое описание числовых данных.
    """
    try:
        return data.describe()
    except Exception as e:
        print(f"Ошибка анализа данных: {e}")
        return None
