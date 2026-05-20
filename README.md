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

### 🔬 Funcionamento e Fundamentação Teórica

#### 1. Camada de Saída: Por que Softmax?
Para a classificação multiclasse mutuamente exclusiva (3 espécies de Iris), a função **Softmax** é aplicada na camada de saída. Ela recebe os valores numéricos brutos da rede (*logits*) e os transforma em uma **distribuição de probabilidade**:
* Todas as probabilidades de saída mapeadas ficam estritamente no intervalo entre $0$ e $1$.
* A soma das três probabilidades de saída é exatamente igual a $1$ ($100\%$).
Isso permite determinar a classe mais provável e exibir uma taxa de confiança legível na interface gráfica.

#### 2. Critérios de Parada do Treinamento
O treinamento do modelo encerra-se ao satisfazer um dos dois parâmetros de parada:
* **Parada por Tolerância de Convergência (Early Stopping)**: Se a diferença absoluta da função de custo (loss) entre a época atual e a anterior for menor do que a tolerância configurada `tol = 1e-5`, assume-se que o modelo convergiu e o processo é interrompido.
* **Limite Máximo de Épocas**: Caso a convergência por variação mínima não ocorra, o treinamento cessa automaticamente ao atingir o limite físico de `max_epochs = 2000` épocas.

#### 3. Como a Rede Aprende?
O aprendizado é supervisionado e ocorre iterativamente através do ciclo:
* **Propagação Direta (Forward Pass)**: O sinal de entrada passa pelas camadas oculta (com ativação Sigmóide) e saída (com ativação Softmax) para produzir as probabilidades preditas.
* **Cálculo da Perda**: Mede-se o erro comparando a distribuição predita com os rótulos reais One-Hot encoded usando a função de **Entropia Cruzada Multiclasse (Categorical Cross-Entropy)**.
* **Retropropagação (Backpropagation)**: Utilizando a **Regra da Cadeia**, os gradientes do erro em relação a cada peso e viés são calculados de trás para frente.
* **Gradiente Descendente**: Os pesos e vieses são ajustados subtraindo o gradiente correspondente multiplicado pela taxa de aprendizado (`learning_rate = 0.1` ou $\eta$), minimizando progressivamente o erro global da rede.

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
