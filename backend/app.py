from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys

# Добавляем текущую директорию в путь импорта
sys.path.append(os.path.dirname(__file__))

from model_loader import load_model_and_predict_for_date

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Включаем CORS для запросов с фронтенда

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    target_date_str = data.get('date')
    
    try:
        # Преобразуем строку даты в объект pd.Period
        target_date = pd.Period(target_date_str, 'M')
        
        # Вызываем функцию для получения матрицы смежности
        adjacency_matrix, graph_data = load_model_and_predict_for_date(target_date)
        
        return jsonify({
            'success': True,
            'graph_data': graph_data,
            'date': target_date.strftime('%Y-%m')
        })
    except Exception as e:
        import traceback
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error: {error_message}")
        print(traceback_str)
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Working directory: {os.getcwd()}")
    
    # Проверяем наличие необходимых файлов
    data_path = os.path.join(os.path.dirname(__file__), '../data/adjacency_matrices.json')
    if os.path.exists(data_path):
        print(f"Data file found: {data_path}")
    else:
        print(f"WARNING: Data file not found: {data_path}")
    
    app.run(debug=True, port=5000)
