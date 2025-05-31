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

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the data
def load_data():
    # Read the CSV file
    data = pd.read_csv('../diabetes.csv')

    # Print input variables
    print("\nVariables de entrada utilizadas:")
    input_variables = data.drop('Diabetes_012', axis=1).columns.tolist()
    for i, var in enumerate(input_variables, 1):
        print(f"{i}. {var}")

    # Separate features and target
    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']

    # Print class distribution before SMOTE
    print("\nDistribución de clases antes del balanceo:")
    print(Counter(y))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Obtener el número de muestras de cada clase
    class_counts = Counter(y_train)
    print("\nNúmero de muestras por clase en el conjunto de entrenamiento:")
    for class_label, count in class_counts.items():
        print(f"Clase {class_label}: {count} muestras")

    # Calcular el número objetivo de muestras (promedio de las clases)
    target_samples = int(sum(class_counts.values()) / 3)
    print(f"\nNúmero objetivo de muestras por clase: {target_samples}")

    # Apply SMOTE with balanced sampling strategy
    sampling_strategy = {
        0: target_samples,  # Reducir la clase mayoritaria
        1: target_samples,  # Balancear la clase de prediabetes
        2: target_samples  # Balancear la clase de diabetes
    }

    print("\nEstrategia de muestreo objetivo:")
    for class_label, n_samples in sampling_strategy.items():
        print(f"Clase {class_label}: {n_samples} muestras")

    # Aplicar undersampling a la clase mayoritaria
    rus = RandomUnderSampler(sampling_strategy={0: target_samples}, random_state=42)
    X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train_scaled, y_train)

    # Aplicar SMOTE para las clases minoritarias
    smote = SMOTE(random_state=42, sampling_strategy={1: target_samples, 2: target_samples})
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_undersampled, y_train_undersampled)

    # Print class distribution after balancing
    print("\nDistribución de clases después del balanceo:")
    print(Counter(y_train_balanced))

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_balanced)
    y_train_tensor = torch.LongTensor(y_train_balanced)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.values)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# Define the neural network architecture
class DiabetesNet(nn.Module):
    def _init_(self, input_size):
        super(DiabetesNet, self)._init_()
        self.layer1 = nn.Linear(input_size, 32)  # Primera capa: 21 -> 32
        self.layer2 = nn.Linear(32, 3)  # Capa de salida: 32 -> 3
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Reducimos el dropout

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x


def plot_confusion_matrix(y_true, y_pred):
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure and axis
        plt.figure(figsize=(10, 8))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes (0)', 'Prediabetes (1)', 'Diabetes (2)'],
                    yticklabels=['No Diabetes (0)', 'Prediabetes (1)', 'Diabetes (2)'])

        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al generar la matriz de confusión: {str(e)}")


def print_metrics(y_true, y_pred):
    try:
        # Calculate metrics for each class
        for i in range(3):
            # True Positives
            tp = np.sum((y_true == i) & (y_pred == i))
            # False Positives
            fp = np.sum((y_true != i) & (y_pred == i))
            # False Negatives
            fn = np.sum((y_true == i) & (y_pred != i))

            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"\nMétricas para la clase {i}:")
            print(f"Precisión: {precision:.2%}")
            print(f"Recuperación (Recall): {recall:.2%}")
    except Exception as e:
        print(f"Error al calcular las métricas: {str(e)}")


def train_model(model, X_train, y_train, X_test, y_test, epochs=200):  # Reducimos las épocas
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Aumentamos el learning rate

    best_accuracy = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
                print(f'Test Accuracy: {accuracy:.4f}')

                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = model.state_dict().copy()
                    print(f'Nuevo mejor modelo encontrado! Accuracy: {accuracy:.4f}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nMejor accuracy alcanzado: {best_accuracy:.4f}')

    return model


def main():
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test = load_data()

        # Initialize the model
        input_size = X_train.shape[1]  # Number of features
        model = DiabetesNet(input_size)

        # Train the model
        print("Starting training...")
        model = train_model(model, X_train, y_train, X_test, y_test)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)

            # Convert tensors to numpy arrays
            y_test_np = y_test.numpy()
            predicted_np = predicted.numpy()

            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test_np, predicted_np))

            # Plot confusion matrix
            print("\nGenerando Matriz de Confusión...")
            plot_confusion_matrix(y_test_np, predicted_np)

            # Print detailed metrics
            print("\nMétricas detalladas por clase:")
            print_metrics(y_test_np, predicted_np)
    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")


if __name__ == "__main__":
    main()