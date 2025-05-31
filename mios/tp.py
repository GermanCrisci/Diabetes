import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the data
def load_data():
    data = pd.read_csv('../diabetes.csv')

    # Filtrar solo clases 0 y 2
    data = data[data['Diabetes_012'].isin([0, 2])]

    print("\nVariables de entrada utilizadas:")
    input_variables = data.drop('Diabetes_012', axis=1).columns.tolist()
    for i, var in enumerate(input_variables, 1):
        print(f"{i}. {var}")

    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']

    print("\nDistribución de clases antes del balanceo:")
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_counts = Counter(y_train)
    print("\nNúmero de muestras por clase en el conjunto de entrenamiento:")
    for class_label, count in class_counts.items():
        print(f"Clase {class_label}: {count} muestras")

    target_samples = min(class_counts.values())  # balancear al menor
    print(f"\nNúmero objetivo de muestras por clase: {target_samples}")

    rus = RandomUnderSampler(sampling_strategy={0: target_samples}, random_state=42)
    X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train_scaled, y_train)

    smote = SMOTE(random_state=42, sampling_strategy={2: target_samples})
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_undersampled, y_train_undersampled)

    print("\nDistribución de clases después del balanceo:")
    print(Counter(y_train_balanced))

    # Convertir etiqueta 2 → 1
    y_train_balanced_binary = y_train_balanced.replace({2: 1})
    y_test_binary = y_test.replace({2: 1})

    X_train_tensor = torch.FloatTensor(X_train_balanced)
    y_train_tensor = torch.LongTensor(y_train_balanced_binary.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test_binary.values)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Define the neural network
def create_diabetes_model(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2)  # 2 clases: 0 (no diabetes), 1 (diabetes)
    )
    return model

def plot_confusion_matrix(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
                    yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al generar la matriz de confusión: {str(e)}")

def print_metrics(y_true, y_pred):
    try:
        for i in range(2): # Asumiendo clases 0 y 1
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"\nMétricas para la clase {i}:")
            print(f"Precisión: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
    except Exception as e:
        print(f"Error al calcular las métricas: {str(e)}")

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Entrenando", unit="época"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
                print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = model.state_dict().copy()
                    print(f"→ Nuevo mejor modelo encontrado con accuracy: {accuracy:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nMejor accuracy alcanzado: {best_accuracy:.4f}")

    return model

def main():
    try:
        X_train, y_train, X_test, y_test = load_data()
        input_size = X_train.shape[1]
        model = create_diabetes_model(input_size) # Cambiado para usar la nueva función

        print("\nIniciando entrenamiento del modelo...")
        model = train_model(model, X_train, y_train, X_test, y_test)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)

            y_test_np = y_test.numpy()
            predicted_np = predicted.numpy()

            print("\nClassification Report:")
            print(classification_report(y_test_np, predicted_np, target_names=["No Diabetes", "Diabetes"], zero_division=0)) # Añadido zero_division

            print("\nGenerando Matriz de Confusión...")
            plot_confusion_matrix(y_test_np, predicted_np)

            print("\nMétricas detalladas por clase:")
            print_metrics(y_test_np, predicted_np)
    except FileNotFoundError:
        print("Error: El archivo 'diabetes.csv' no se encontró. Asegúrate de que esté en el mismo directorio que el script.")
    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()