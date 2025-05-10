def train_test_split_time_based(data, test_size=0.2):
    """
    Разделяет данные на тренировочные и тестовые по времени
    """
    n_samples = len(data)
    split_idx = int(n_samples * (1 - test_size))
    return data[:split_idx], data[split_idx:]
