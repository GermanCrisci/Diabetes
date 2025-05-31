import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# --- Configuración Global ---
SEED = 19982809
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
# Updated CLASS_NAMES to reflect the 3 original classes
CLASS_NAMES = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
DATA_FILE = '../diabetes_menos_columnas.csv'

# --- Configuración de Pesos de Clase ---
# Si MANUAL_CLASS_WEIGHTS es True, se usarán los valores de CLASS_WEIGHTS_VALUES.
# De lo contrario, los pesos se calcularán automáticamente con compute_class_weight.
MANUAL_CLASS_WEIGHTS = False
# CLASS_WEIGHTS_VALUES should now have 3 elements if manually set
CLASS_WEIGHTS_VALUES = [1.0, 1.0, 1.0] # Placeholder, will be computed automatically if MANUAL_CLASS_WEIGHTS is False

# --- Configuración de Umbral de Decisión (NUEVO) ---
# En un modelo multiclass, la predicción por defecto es la clase con la probabilidad más alta (argmax).
# Este umbral ya no se usa directamente para la clasificación final en `evaluate_model`
# pero se mantiene como una configuración global si se desea una lógica de decisión personalizada.
PREDICTION_THRESHOLD = 0.50

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- CUDA Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed_all(SEED) # for reproducibility on CUDA

# --- Definición del Modelo ---
class DiabetesClassifier(nn.Module):
    """
    Red Neuronal para clasificación multiclass de diabetes (3 clases).
    Arquitectura mejorada con más capas y regularización.
    """
    def __init__(self, input_size):
        super(DiabetesClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout ajustado
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout ajustado
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3) # Reverted to 3 for multiclass classification
        )

    def forward(self, x):
        return self.model(x)

# --- Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(batch_size: int = 64):
    """
    Carga, filtra, preprocesa y balancea los datos del archivo CSV.
    Calcula pesos de clase basados en la distribución original para CrossEntropyLoss.

    Args:
        batch_size (int): Tamaño del lote para el DataLoader.

    Returns:
        tuple: (train_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor)
               Retorna None para los primeros elementos si el archivo no se encuentra.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Error: El archivo '{DATA_FILE}' no se encontró. Asegúrate de que esté en el mismo directorio que el script."
        )

    data = pd.read_csv(DATA_FILE)

    print("\nVariables de entrada utilizadas:")
    input_variables = data.drop('Diabetes_012', axis=1).columns.tolist()
    for i, var in enumerate(input_variables, 1):
        print(f"{i}. {var}")

    # --- No binarize the target variable; keep original 0, 1, 2 classes ---
    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']

    print("\nDistribución de clases original:")
    print(Counter(y))

    # train_test_split con stratify sobre las etiquetas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    input_size = X_train_scaled.shape[1]

    print("\nDistribución de clases en entrenamiento ANTES del balanceo:")
    print(Counter(y_train))

    # --- Cálculo de pesos de clase (manual o automático) ---
    if MANUAL_CLASS_WEIGHTS:
        # Ensure CLASS_WEIGHTS_VALUES has 3 elements for 3-class classification
        if len(CLASS_WEIGHTS_VALUES) != 3:
            raise ValueError("CLASS_WEIGHTS_VALUES must have 3 elements for 3-class classification.")
        class_weights_tensor = torch.FloatTensor(CLASS_WEIGHTS_VALUES)
        print(f"\nUsando pesos de clase manuales: {class_weights_tensor}")
    else:
        # Cálculo de pesos de clase basado en la distribución original de y_train
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train), # Will now be [0.0, 1.0, 2.0]
            y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights_array)
        print(f"\nPesos de clase calculados automáticamente: {class_weights_tensor}")

    # SMOTE para balancear el conjunto de entrenamiento
    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced_smote = smote.fit_resample(X_train_scaled, y_train)
    print("\nDistribución de clases en entrenamiento DESPUÉS de SMOTE:")
    print(Counter(y_train_balanced_smote))

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.FloatTensor(X_train_balanced)
    y_train_tensor = torch.LongTensor(y_train_balanced_smote.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))

    return train_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor

# --- Funciones de Evaluación y Visualización ---
# NOTE: The 'threshold' parameter is kept but not used for prediction in this multiclass setup.
# Predictions are based on argmax.
def evaluate_model(model: nn.Module, X_data: torch.Tensor, y_true: torch.Tensor, device: torch.device, threshold: float = 0.5):
    """
    Evalúa el modelo en un conjunto de datos dado.
    Para modelos multiclass, la predicción se basa en argmax.
    Las métricas de precisión, recall y F1-score se calculan para la clase "Diabetes" (clase 2).

    Args:
        model (nn.Module): El modelo a evaluar.
        X_data (torch.Tensor): Tensores de características del conjunto de datos.
        y_true (torch.Tensor): Tensores de etiquetas reales del conjunto de datos.
        device (torch.device): El dispositivo (CPU/CUDA) donde está el modelo y los datos.
        threshold (float): Este parámetro no se usa para la predicción final en este contexto multiclass.

    Returns:
        tuple: (y_pred_np, accuracy, f1_score_diabetes, precision_diabetes, recall_diabetes, y_true_np)
               Retorna métricas clave y las predicciones/valores reales como arrays de numpy.
    """
    model.eval()
    with torch.no_grad():
        X_data_dev = X_data.to(device)
        outputs_dev = model(X_data_dev)

        # For multiclass, prediction is typically the class with the highest probability
        predicted_dev = torch.argmax(outputs_dev, dim=1)

        y_true_np = y_true.cpu().numpy()
        predicted_np = predicted_dev.cpu().numpy()

        accuracy = accuracy_score(y_true_np, predicted_np)
        # Metrics for the 'Diabetes' class (which is class 2)
        # Use average=None to get per-class scores, then select the score for class 2
        precision_all_classes = precision_score(y_true_np, predicted_np, average=None, zero_division=0)
        recall_all_classes = recall_score(y_true_np, predicted_np, average=None, zero_division=0)
        f1_all_classes = f1_score(y_true_np, predicted_np, average=None, zero_division=0)

        precision_diabetes = precision_all_classes[2]
        recall_diabetes = recall_all_classes[2]
        f1_diabetes = f1_all_classes[2]

    return predicted_np, accuracy, f1_diabetes, precision_diabetes, recall_diabetes, y_true_np


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
    """
    Genera y muestra una matriz de confusión.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al generar la matriz de confusión: {str(e)}")

def plot_metrics(train_losses, test_f1_scores_diabetes, test_accuracies):
    """
    Genera gráficos de la pérdida de entrenamiento y métricas de prueba.
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico de Pérdida de Entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Pérdida de Entrenamiento')
    plt.title('Pérdida de Entrenamiento por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.legend()

    # Gráfico de Métricas de Prueba (F1-Score Diabetes y Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_f1_scores_diabetes, label='F1-Score (Diabetes) en Test', color='orange')
    plt.plot(epochs_range, test_accuracies, label='Accuracy en Test', color='green', linestyle='--')
    plt.title('Métricas de Test por Época')
    plt.xlabel('Época')
    plt.ylabel('Valor de Métrica')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- Función de Entrenamiento ---
def train_model(model: nn.Module, train_loader: DataLoader, X_test_tensor: torch.Tensor,
                y_test_tensor: torch.Tensor, device: torch.device, class_weights: torch.Tensor = None,
                epochs: int = 100, lr: float = 0.001):
    """
    Entrena el modelo de red neuronal.

    Args:
        model (nn.Module): El modelo a entrenar.
        train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.
        X_test_tensor (torch.Tensor): Tensores de características del conjunto de prueba.
        y_test_tensor (torch.Tensor): Tensores de etiquetas reales del conjunto de prueba (en CPU).
        device (torch.device): El dispositivo (CPU/CUDA) donde entrenar el modelo.
        class_weights (torch.Tensor, optional): Pesos de clase para CrossEntropyLoss. Por defecto None.
        epochs (int): Número de épocas de entrenamiento.
        lr (float): Tasa de aprendizaje inicial.

    Returns:
        nn.Module: El modelo entrenado con el mejor rendimiento en el conjunto de prueba.
        list: Lista de pérdidas de entrenamiento por época.
        list: Lista de F1-Scores (Diabetes) en el conjunto de prueba por época.
        list: Lista de Accuracies en el conjunto de prueba por época.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Usando pesos de clase en CrossEntropyLoss: {class_weights} en dispositivo {class_weights.device}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("No se usan pesos de clase en CrossEntropyLoss.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Monitoreamos el F1-Score para la clase 'Diabetes' (clase 2) para el scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

    best_f1_diabetes = 0
    best_model_state = None
    epochs_no_improve = 0
    early_stopping_patience = 20

    train_losses = []
    test_accuracies = []
    test_f1_scores_diabetes = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for data_batch, targets_batch in batch_iterator:
            data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data_batch.size(0)
            batch_iterator.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluación en el conjunto de prueba (predicción por argmax)
        _, accuracy, f1_diabetes, precision_diabetes, recall_diabetes, _ = \
            evaluate_model(model, X_test_tensor, y_test_tensor, device, threshold=PREDICTION_THRESHOLD) # Threshold is ignored for prediction here

        test_accuracies.append(accuracy)
        test_f1_scores_diabetes.append(f1_diabetes)

        print(f"\nEpoch [{epoch + 1}/{epochs}], Avg Train Loss: {epoch_loss:.4f}, "
              f"Test Acc: {accuracy:.4f}, Test Prec (D): {precision_diabetes:.4f}, "
              f"Test Recall (D): {recall_diabetes:.4f}, Test F1 (D): {f1_diabetes:.4f}")

        # Paso del scheduler basado en el F1-Score de la clase Diabetes
        scheduler.step(f1_diabetes)

        # Lógica de Early Stopping
        if f1_diabetes > best_f1_diabetes:
            best_f1_diabetes = f1_diabetes
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"→ Nuevo mejor modelo (F1-Score Diabetes): {f1_diabetes:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping activado después de {early_stopping_patience} épocas sin mejora en F1-Score (Diabetes).")
                break # Sale del bucle de entrenamiento

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\nEntrenamiento finalizado. Mejor F1-Score (Diabetes) en test alcanzado: {best_f1_diabetes:.4f}")
    else:
        print("\nEntrenamiento finalizado. No se guardó ningún mejor modelo (revisar configuración).")
    return model, train_losses, test_f1_scores_diabetes, test_accuracies

# --- Función Principal ---
def main():
    try:
        train_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor = \
            load_and_preprocess_data(batch_size=BATCH_SIZE)

        model = DiabetesClassifier(input_size).to(device)

        print("\nIniciando entrenamiento del modelo...")
        trained_model, train_losses, test_f1_scores_diabetes, test_accuracies = train_model(
            model, train_loader, X_test_tensor, y_test_tensor, device,
            class_weights=class_weights_tensor, epochs=EPOCHS, lr=LEARNING_RATE
        )

        print(f"\nEvaluando el mejor modelo en el conjunto de prueba final (predicción por argmax)...")
        y_pred_final, accuracy_final, f1_diabetes_final, precision_diabetes_final, recall_diabetes_final, y_true_final = \
            evaluate_model(trained_model, X_test_tensor, y_test_tensor, device, threshold=PREDICTION_THRESHOLD) # Threshold ignored for prediction

        print("\n--- Resultados Finales ---")
        print("\nClassification Report:")
        # The classification report will show metrics for all 3 classes (0, 1, 2)
        print(classification_report(y_true_final, y_pred_final, target_names=CLASS_NAMES, zero_division=0))

        print("\nMétricas detalladas por clase (sklearn.metrics):")
        # Get all per-class metrics
        prec_all_classes = precision_score(y_true_final, y_pred_final, average=None, zero_division=0)
        rec_all_classes = recall_score(y_true_final, y_pred_final, average=None, zero_division=0)
        f1_all_classes = f1_score(y_true_final, y_pred_final, average=None, zero_division=0)

        print(f"Métricas para la clase '{CLASS_NAMES[0]}' (clase 0):")
        print(f"Precisión: {prec_all_classes[0]:.2%}")
        print(f"Recall: {rec_all_classes[0]:.2%}")
        print(f"F1-Score: {f1_all_classes[0]:.2%}")

        print(f"\nMétricas para la clase '{CLASS_NAMES[1]}' (clase 1):")
        print(f"Precisión: {prec_all_classes[1]:.2%}")
        print(f"Recall: {rec_all_classes[1]:.2%}")
        print(f"F1-Score: {f1_all_classes[1]:.2%}")

        print(f"\nMétricas para la clase '{CLASS_NAMES[2]}' (clase 2 - Diabetes):")
        # These are already calculated in evaluate_model, so just print them
        print(f"Precisión: {precision_diabetes_final:.2%}")
        print(f"Recall: {recall_diabetes_final:.2%}")
        print(f"F1-Score: {f1_diabetes_final:.2%}")

        print("\nGenerando Matriz de Confusión...")
        plot_confusion_matrix(y_true_final, y_pred_final, class_names=CLASS_NAMES)

        print("\nGenerando gráficos de entrenamiento...")
        plot_metrics(train_losses, test_f1_scores_diabetes, test_accuracies)

        print("\nProceso finalizado.")

    except FileNotFoundError as e:
        print(e)
        print("Asegúrate de que el archivo 'diabetes_menos_columnas.csv' esté en el directorio correcto.")
    except Exception as e:
        print(f"Error fatal en la ejecución principal: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()