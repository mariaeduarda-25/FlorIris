import os
import sys
import pickle
import numpy as np

# Adicionar o diretório do projeto ao path para importação
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Carregar o dataset pré-processado
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_processed.npz')
dataset = np.load(DATA_PATH, allow_pickle=True)
X_full = dataset['X']
y_full = dataset['y']
X_min = dataset['X_min']
X_max = dataset['X_max']
species_map = dataset['species_map']

class MLPTrainer3Neurons:
    """Implementação dedicada do MLP com 3 neurônios para exportação."""
    
    def __init__(self, n_input=4, n_hidden=3, n_output=3, learning_rate=0.1, max_epochs=2000, tol=1e-5):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.epochs_trained = 0
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.X_min = X_min
        self.X_max = X_max
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def train(self, X_train, y_train, X_test, y_test, seed):
        # Configurar reprodutibilidade de inicialização de pesos baseada na semente
        np.random.seed(seed + 1000)
        self.W1 = np.random.randn(self.n_input, self.n_hidden) * np.sqrt(2.0 / self.n_input)
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, self.n_output) * np.sqrt(2.0 / self.n_hidden)
        self.b2 = np.zeros((1, self.n_output))
        
        prev_loss = float('inf')
        
        for epoch in range(1, self.max_epochs + 1):
            # Forward pass
            z1 = np.dot(X_train, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.softmax(z2)
            
            # Cross-entropy loss
            loss = -np.mean(np.sum(y_train * np.log(a2 + 1e-8), axis=1))
            
            # Backpropagation
            m = X_train.shape[0]
            dz2 = a2 - y_train
            dW2 = np.dot(a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.sigmoid_derivative(a1)
            dW1 = np.dot(X_train.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m
            
            # Gradient descent update
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            
            # Early stopping
            if abs(prev_loss - loss) < self.tol:
                self.epochs_trained = epoch
                break
            prev_loss = loss
        else:
            self.epochs_trained = self.max_epochs
            
        self.train_accuracy = self._accuracy(X_train, y_train)
        self.test_accuracy = self._accuracy(X_test, y_test)
        
        return self.epochs_trained

    def _accuracy(self, X_norm, y):
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        predictions = np.argmax(a2, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels) * 100.0

def main():
    seed = 202
    print(f"Iniciando treinamento com semente {seed} (3 neurônios na camada oculta)...")
    
    # Garantir split de treino/teste reprodutível usando o seed 202
    np.random.seed(seed)
    indices = np.random.permutation(len(X_full))
    split = int(0.7 * len(X_full))
    
    X_train = X_full[indices[:split]]
    y_train = y_full[indices[:split]]
    X_test = X_full[indices[split:]]
    y_test = y_full[indices[split:]]
    
    # Instanciar e treinar o MLP de 3 neurônios
    trainer = MLPTrainer3Neurons(n_hidden=3)
    epochs = trainer.train(X_train, y_train, X_test, y_test, seed)
    
    print("\nTreinamento Concluído:")
    print(f"  - Épocas treinadas: {epochs}")
    print(f"  - Acurácia no conjunto de Treino: {trainer.train_accuracy:.2f}%")
    print(f"  - Acurácia no conjunto de Teste: {trainer.test_accuracy:.2f}%")
    
    # Calcular Métricas adicionais para validação
    z1 = np.dot(X_test, trainer.W1) + trainer.b1
    a1 = trainer.sigmoid(z1)
    z2 = np.dot(a1, trainer.W2) + trainer.b2
    a2 = trainer.softmax(z2)
    mse_test = np.mean(np.sum((a2 - y_test) ** 2, axis=1))
    print(f"  - Erro Quadrático Médio (MSE) no Teste: {mse_test:.6f}")
    
    # Montar dicionário de dados a serem salvos
    model_data = {
        'W1': trainer.W1,
        'b1': trainer.b1,
        'W2': trainer.W2,
        'b2': trainer.b2,
        'X_min': trainer.X_min,
        'X_max': trainer.X_max,
        'species_map': species_map
    }
    
    # Salvar em mlp_model.pkl
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nPesos salvos com sucesso em: {output_path}")
    print(f"Estrutura do arquivo de pesos:")
    for k, v in model_data.items():
        shape_info = getattr(v, 'shape', type(v))
        print(f"  - {k}: formato {shape_info}")

if __name__ == "__main__":
    main()
