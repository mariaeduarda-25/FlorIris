# Projeto de IA: Classificador de Flores Iris com MLP

Este projeto foi desenvolvido para a disciplina de Inteligência Artificial, utilizando uma rede neural **Perceptron Multi Camadas (MLP)** implementada do zero (sem bibliotecas de alto nível como scikit-learn) para classificar espécies de flores Iris.

## 🚀 Tecnologias Utilizadas

- **Python 3**: Linguagem base.
- **NumPy**: Para operações matemáticas e álgebra linear.
- **Pandas**: Para manipulação inicial dos dados.
- **Django**: Framework web para a interface do usuário.
- **KaggleHub**: Para download automático do dataset oficial.

## 🧠 Detalhes da Implementação de IA

- **Modelo**: Perceptron Multi Camadas (MLP).
- **Arquitetura**:
  - Camada de Entrada: 4 neurônios (medidas das sépalas e pétalas).
  - Camada Oculta: 50 neurônios (conforme requisito).
  - Camada de Saída: 3 neurônios (uma para cada espécie).
- **Funções de Ativação**:
  - Camada Oculta: Sigmóide.
  - Camada de Saída: Softmax (para classificação multi-classe).
- **Treinamento**:
  - Divisão de Dados: 70% para treino e 30% para teste.
  - Otimização: Gradiente Descendente.
  - Normalização: Min-Max Scaling aplicada aos dados de entrada.

## 📂 Estrutura do Projeto

- `mlp_implementation.py`: Contém a classe MLP e o script de treinamento.
- `mlp_model.pkl`: Arquivo com os pesos treinados e parâmetros de normalização.
- `iris_project/`: Diretório do projeto Django.
  - `predictor/`: App Django que gerencia a lógica de predição.
  - `templates/index.html`: Interface web elegante e responsiva.

## 🛠️ Como Executar

1. Instale as dependências:
   ```bash
   pip install django numpy pandas kagglehub
   ```

2. (Opcional) Treine o modelo novamente:
   ```bash
   python mlp_implementation.py
   ```

3. Inicie o servidor Django:
   ```bash
   cd iris_project
   python manage.py runserver
   ```

4. Acesse no navegador: `http://127.0.0.1:8000`

## 📊 Dataset
O projeto utiliza o [Iris Flower Dataset](https://www.kaggle.com/arshid/iris-flower-dataset) do Kaggle, que contém 150 amostras de 3 espécies de Iris (Setosa, Versicolor e Virginica).
