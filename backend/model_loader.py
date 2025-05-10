import os
import json
import pandas as pd
import numpy as np
import torch
import torch.serialization
from sklearn.preprocessing import MinMaxScaler
import numpy.core.multiarray

# Добавляем необходимые классы в список безопасных
torch.serialization.add_safe_globals([
    MinMaxScaler, 
    numpy.core.multiarray._reconstruct
])

# Импортируем напрямую из файла
from SparseGraphPredictorModel import SparseGraphPredictorModel, train_test_split_time_based

# Функция для оценки различия между матрицами
def matrix_difference(matrix1, matrix2):
    """Вычисляет среднюю абсолютную разницу между двумя матрицами"""
    return np.mean(np.abs(matrix1 - matrix2))

# Патчим метод predict_next_month для предотвращения переполнения
original_predict_next_month = SparseGraphPredictorModel.predict_next_month

def safe_predict_next_month(self):
    """
    Безопасная версия метода predict_next_month, предотвращающая переполнение
    при вычислении экспоненты
    """
    last_5_months = self.processed_data[-self.seq_length:]
    inputs = torch.tensor(last_5_months, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    self.model.eval()
    with torch.no_grad():
        month_indices = torch.tensor([m % 12 for m in range(len(last_5_months))], dtype=torch.long).to(self.device)
        outputs = self.model(inputs, month_indices)
    
    predicted_matrix = outputs.cpu().numpy()[0]
    predicted_matrix_flat = predicted_matrix.reshape(1, -1)
    predicted_matrix_scaled = self.scaler.inverse_transform(predicted_matrix_flat)
    
    max_safe_value = 20
    predicted_matrix_scaled = np.clip(predicted_matrix_scaled, -max_safe_value, max_safe_value)
    
    predicted_matrix_exp = np.exp(predicted_matrix_scaled.reshape(self.num_nodes, self.num_nodes)) - self.eps
    predicted_matrix_exp[predicted_matrix_exp < 0] = 0
    
    return predicted_matrix_exp

SparseGraphPredictorModel.predict_next_month = safe_predict_next_month

def load_model_and_predict_for_date(target_date):
    """
    Загружает модель и предсказывает матрицу смежности на указанную дату
    """
    print(f"\n=== Предсказание для даты: {target_date} ===")
    
    data_path = os.path.join(os.path.dirname(__file__), '../data/adjacency_matrices.json')
    with open(data_path, 'r') as f:
        loaded_data = json.load(f)
    
    adjacency_matrices = {
        pd.Period(period_str, 'M'): np.array(matrix)
        for period_str, matrix in loaded_data.items()
    }
    
    model = SparseGraphPredictorModel(
        adjacency_matrices=adjacency_matrices,
        seq_length=5,
        hidden_size=256
    )
    
    model.load_data().preprocess_data().build_model()
    model_path = os.path.join(os.path.dirname(__file__), '../models/model_checkpoint.pth')
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Ошибка при загрузке с weights_only=True: {e}")
            print("Пробуем загрузить с weights_only=False (это безопасно, если вы доверяете файлу контрольной точки)")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler = checkpoint.get('scaler')
        model.eps = checkpoint.get('eps', 1e-8)
        model.num_nodes = checkpoint.get('num_nodes')
        model.model.eval()
    
    last_known_date = max(adjacency_matrices.keys())
    print(f"Последняя известная дата: {last_known_date}")
    
    if target_date in adjacency_matrices:
        print(f"Используем существующую матрицу для {target_date}")
        matrix = adjacency_matrices[target_date]

    elif target_date > last_known_date:
        print(f"Прогнозируем матрицу для будущей даты: {target_date}")
        current_matrices = adjacency_matrices.copy()
        current_date = last_known_date
        prev_matrix = current_matrices[current_date]

        while current_date < target_date:
            model.adjacency_matrices = current_matrices
            model.load_data().preprocess_data()
            
            next_date = current_date + 1
            print(f"Прогнозируем для даты: {next_date}")
            
            next_matrix = model.predict_next_month()
            
            # Проверяем разницу между матрицами
            diff = matrix_difference(prev_matrix, next_matrix)
            print(f"Разница между матрицами {current_date} и {next_date}: {diff:.6f}")
            
            # Если матрицы слишком похожи, добавляем случайный шум
            if diff < 0.001:
                print(f"Предупреждение: Матрицы почти идентичны, добавляем шум")
                noise = np.random.normal(0, 0.1, next_matrix.shape)
                next_matrix = next_matrix + noise
                next_matrix[next_matrix < 0] = 0
            
            current_matrices[next_date] = next_matrix
            current_date = next_date
            prev_matrix = next_matrix
        
        matrix = current_matrices[target_date]
    else:
        raise ValueError(f"Дата {target_date} находится раньше последних известных данных")
    
    # Проверяем, что матрица не пустая
    if np.max(matrix) < 1e-6:
        print("Предупреждение: Матрица содержит только очень маленькие значения!")
    
    # Выводим базовую статистику
    print(f"Статистика матрицы для {target_date}:")
    print(f"  Min: {np.min(matrix):.6f}, Max: {np.max(matrix):.6f}, Mean: {np.mean(matrix):.6f}")
    print(f"  Ненулевых элементов: {np.sum(matrix > 0.05)}")
    
    # Преобразуем матрицу в данные для графа
    graph_data = matrix_to_graph_data(matrix)
    print(f"Создан граф с {len(graph_data['nodes'])} узлами и {len(graph_data['links'])} связями")
    
    return matrix, graph_data

def matrix_to_graph_data(matrix, threshold_percentile=90, max_links=150):
    """
    Преобразует матрицу смежности в формат для визуализации графа,
    отображая только самые сильные связи
    """
    n_nodes = matrix.shape[0]
    
    label_map = {
        0: 'math.AC', 1: 'math.AG', 2: 'math.AP', 3: 'math.AT', 4: 'math.CA', 
        5: 'math.CO', 6: 'math.CT', 7: 'math.CV', 8: 'math.DG', 9: 'math.DS', 
        10: 'math.FA', 11: 'math.GM', 12: 'math.GN', 13: 'math.GR', 14: 'math.GT', 
        15: 'math.HO', 16: 'math.IT', 17: 'math.KT', 18: 'math.LO', 19: 'math.MG', 
        20: 'math.MP', 21: 'math.NA', 22: 'math.NT', 23: 'math.OA', 24: 'math.OC', 
        25: 'math.PR', 26: 'math.QA', 27: 'math.RA', 28: 'math.RT', 29: 'math.SG', 
        30: 'math.SP', 31: 'math.ST'
    }
    
    nodes = [{"id": i, "name": label_map.get(i, f"Node {i+1}")} for i in range(n_nodes)]
    all_links = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if matrix[i, j] > 0:
                all_links.append({
                    "source": i,
                    "target": j,
                    "weight": float(matrix[i, j])
                })
    
    # Сортируем связи по весу (от большего к меньшему)
    all_links.sort(key=lambda x: x["weight"], reverse=True)
    
    # Определяем порог, используя процентиль
    if len(all_links) > 0:
        weights = [link["weight"] for link in all_links]
        threshold = np.percentile(weights, threshold_percentile)
        
        # Выбираем связи выше порога
        filtered_links = [link for link in all_links if link["weight"] >= threshold]
        if len(filtered_links) > max_links:
            filtered_links = filtered_links[:max_links]
            
        print(f"Порог силы связи: {threshold:.6f}")
        print(f"Отображаем {len(filtered_links)} из {len(all_links)} связей")
    else:
        filtered_links = []
    
    return {
        "nodes": nodes,
        "links": filtered_links
    }
