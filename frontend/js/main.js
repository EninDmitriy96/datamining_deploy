document.addEventListener('DOMContentLoaded', function() {
    const dateInput = document.getElementById('date-input');
    const predictBtn = document.getElementById('predict-btn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const resultDate = document.getElementById('result-date');
    const graphContainer = document.getElementById('graph-container');
    
    // Устанавливаем текущую дату по умолчанию
    const now = new Date();
    dateInput.value = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
    
    predictBtn.addEventListener('click', async function() {
        const targetDate = dateInput.value;
        
        // Показываем loader, скрываем ошибки
        loading.classList.remove('hidden');
        error.classList.add('hidden');
        resultDate.textContent = '-';
        graphContainer.innerHTML = '';
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ date: targetDate }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                resultDate.textContent = data.date;
                drawGraph(data.graph_data, graphContainer);
            } else {
                error.textContent = data.error || 'Ошибка при получении предсказания';
                error.classList.remove('hidden');
            }
        } catch (err) {
            error.textContent = `Ошибка: ${err.message}`;
            error.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });
});
