import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class AdjacencyMatrixDataset(Dataset):
    def __init__(self, data, months, seq_length=5):
        self.data = data
        self.months = months
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        month_indices = self.months[idx:idx+self.seq_length]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(month_indices, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32)
        )


class SparseGraphPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_nodes, num_months=12, embed_dim=8):
        super(SparseGraphPredictor, self).__init__()
        self.num_nodes = num_nodes
        self.month_embedding = nn.Embedding(num_embeddings=num_months, embedding_dim=embed_dim)
        total_input_size = input_size + seq_length * embed_dim

        self.model = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_nodes * num_nodes)
        )

    def forward(self, x, months):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x_flat = x.view(batch_size, -1)
        month_emb = self.month_embedding(months)
        month_emb_flat = month_emb.view(batch_size, -1)
        combined = torch.cat([x_flat, month_emb_flat], dim=1)
        out = self.model(combined)
        out = out.view(batch_size, self.num_nodes, self.num_nodes)
        return out


class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.1, weight_factor=10):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight_factor = weight_factor
        
    def forward(self, inputs, targets):
        weights = torch.ones_like(targets)
        non_zero_mask = targets > self.threshold
        weights[non_zero_mask] = self.weight_factor
        loss = torch.mean(weights * (inputs - targets) ** 2)
        return loss


class SparseGraphPredictorModel:
    def __init__(self, adjacency_matrices, seq_length=5, hidden_size=256, test_size=0.2):
        self.adjacency_matrices = adjacency_matrices
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.test_size = test_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.eps = 1e-8
        self.model = None
        self.best_test_loss = float('inf')  # Инициализируем бесконечностью
        self.best_model_path = 'best_model.pth'

    def load_data(self):
        sorted_months = sorted(self.adjacency_matrices.keys())
        self.data_months = sorted_months
        matrices_list = [self.adjacency_matrices[month] for month in sorted_months]
        self.data = np.array(matrices_list)
        return self

    def preprocess_data(self):
        eps = self.eps
        log_data = np.log(self.data + eps)
        self.scaler = MinMaxScaler()
        n_samples, n_nodes, _ = log_data.shape
        log_data_flat = log_data.reshape(n_samples, -1)
        scaled_data = self.scaler.fit_transform(log_data_flat)
        self.processed_data = scaled_data.reshape(n_samples, n_nodes, n_nodes)
        self.num_nodes = n_nodes
        return self

    def split_data(self):
        train_data, test_data = train_test_split_time_based(self.processed_data, self.test_size)
        self.train_data = train_data
        self.test_data = test_data
        
        month_indices = [int(month.strftime('%m')) - 1 for month in self.data_months]
        self.train_months = month_indices[:len(train_data)]
        self.test_months = month_indices[len(train_data):]
        return self

    def setup_datasets_and_loaders(self):
        self.train_dataset = AdjacencyMatrixDataset(self.train_data, self.train_months, self.seq_length)
        self.test_dataset = AdjacencyMatrixDataset(self.test_data, self.test_months, self.seq_length)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        return self

    def build_model(self):
        input_size = self.seq_length * self.num_nodes * self.num_nodes
        self.model = SparseGraphPredictor(
            input_size, self.hidden_size, self.seq_length,
            self.num_nodes, num_months=12, embed_dim=8
        ).to(self.device)
        self.criterion = WeightedMSELoss(threshold=0.05, weight_factor=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        return self

    def train(self, num_epochs=200):
        self.model.train()
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        self.best_model_path = 'best_model.pth'
    
        pbar = tqdm(range(num_epochs), desc="Training", total=num_epochs)
        for epoch in pbar:
            total_loss = 0
            self.model.train()
    
            for inputs, months, targets in self.train_loader:
                inputs, months, targets = inputs.to(self.device), months.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs, months)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
            train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(train_loss)
    
            test_loss = self.evaluate()
            self.test_losses.append(test_loss)
    
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.save_model(self.best_model_path)
    
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.6f}',
                'Test Loss': f'{test_loss:.6f}',
                'Best Test Loss': f'{self.best_test_loss:.6f}'
            })
    
        # После обучения загружаем лучшую модель
        self.load_best_model()
        return self

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, months, targets in self.test_loader:
                inputs, months, targets = inputs.to(self.device), months.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, months)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss

    def save_model(self, path='sparse_graph_model.pth'):
        torch.save(self.model.state_dict(), path)
        return self

    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.model.to(self.device)
            self.model.eval()
        else:
            print("Лучшая модель не найдена.")
        return self

    def save_checkpoint(self, path='model_checkpoint.pth'):
        """
        Сохраняет весь объект класса вместе с моделью, скалером и параметрами.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'eps': self.eps,
            'num_nodes': self.num_nodes,
            'seq_length': self.seq_length,
            'hidden_size': self.hidden_size,
            'best_test_loss': self.best_test_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'data_months': self.data_months,
        }
        torch.save(checkpoint, path)
        return self

    @classmethod
    def load_checkpoint(cls, path='model_checkpoint.pth', adjacency_matrices=None):
        """
        Восстанавливает объект из файла. adjacency_matrices требуется для повторной инициализации.
        """
        if adjacency_matrices is None:
            raise ValueError("Необходимо передать adjacency_matrices при загрузке модели.")

        checkpoint = torch.load(path)

        # Создаем новый экземпляр класса с минимальными параметрами
        instance = cls(
            adjacency_matrices=adjacency_matrices,
            seq_length=checkpoint['seq_length'],
            hidden_size=checkpoint['hidden_size'],
            test_size=0.2
        )

        # Восстанавливаем данные
        instance.load_data().preprocess_data()
        instance.scaler = checkpoint['scaler']
        instance.eps = checkpoint['eps']
        instance.num_nodes = checkpoint['num_nodes']
        instance.data_months = checkpoint['data_months']
        instance.build_model()

        # Загружаем состояние модели и оптимизатора
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        instance.train_losses = checkpoint['train_losses']
        instance.test_losses = checkpoint['test_losses']
        instance.best_test_loss = checkpoint['best_test_loss']

        instance.model.to(instance.device)
        instance.model.eval()

        return instance

    def predict_next_month(self):
        last_5_months = self.processed_data[-self.seq_length:]
        inputs = torch.tensor(last_5_months, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, torch.tensor([m % 12 for m in range(len(last_5_months))], dtype=torch.long).to(self.device))
        
        predicted_matrix = outputs.cpu().numpy()[0]
        predicted_matrix_flat = predicted_matrix.reshape(1, -1)
        predicted_matrix_scaled = self.scaler.inverse_transform(predicted_matrix_flat)
        predicted_matrix_exp = np.exp(predicted_matrix_scaled.reshape(self.num_nodes, self.num_nodes)) - self.eps
        predicted_matrix_exp[predicted_matrix_exp < 0] = 0
        return predicted_matrix_exp

    def run(self, num_epochs=200):
        self.load_data().preprocess_data().split_data().setup_datasets_and_loaders().build_model()
        self.train(num_epochs).save_model()
        predicted_matrix = self.predict_next_month()
        return self.model, predicted_matrix


def train_test_split_time_based(data, test_size=0.2):
    n_samples = len(data)
    split_idx = int(n_samples * (1 - test_size))
    return data[:split_idx], data[split_idx:]