import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
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
SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # 20% de los datos de entrenamiento para validación
CLASS_NAMES = ["No Diabetes", "Diabetes"]
DATA_FILE = 'diabetes.csv'

# --- Variables Globales para Guardar el Modelo y Gráficos ---
MODEL_SAVE_DIR = 'trained_models_convnet'  # Adjusted directory name for ConvNet
MODEL_NAME = 'diabetes_convnet_model.pth'  # Adjusted model name
CONFUSION_MATRIX_NAME = 'convnet_validation_confusion_matrix.png'  # Name for validation confusion matrix
TRAINING_METRICS_NAME = 'convnet_training_metrics.png'  # Name for training metrics plot
FINAL_CONFUSION_MATRIX_NAME = 'convnet_final_test_confusion_matrix.png'  # Name for final test confusion matrix
FINAL_CLASSIFICATION_REPORT_NAME = 'convnet_final_classification_report.txt'  # Name for final classification report

# --- Variables Globales para los Nombres de Archivos CSV ---
FINAL_TEST_CSV_NAME = 'diabetes_convnet_prueba_final.csv'
TRAIN_VAL_CSV_NAME = 'diabetes_convnet_entrenamiento_validacion.csv'

# --- Configuración de Pesos de Clase ---
MANUAL_CLASS_WEIGHTS = True
CLASS_WEIGHTS_VALUES = [1.0, 1.5]  # Ajustado a 2 clases

# --- Configuración de Umbral de Decisión ---
PREDICTION_THRESHOLD = 0.20

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- CUDA Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed_all(SEED)  # for reproducibility on CUDA


# --- Función para Verificar y Crear Carpetas ---
def ensure_directories_exist(directories: list):
    """
    Verifica si una lista de directorios existe y los crea si no es así.

    Args:
        directories (list): Una lista de rutas de directorios a verificar/crear.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directorio creado: {directory}")
        else:
            print(f"Directorio ya existe: {directory}")


# --- Definición del Modelo Convolucional ---
class DiabetesConvNet(nn.Module):
    """
    Red Neuronal Convolucional para clasificación binaria de diabetes.
    La entrada se reorganiza en una matriz 2D para procesamiento convolucional.
    """

    def __init__(self, input_size):
        super(DiabetesConvNet, self).__init__()

        # Calcular dimensiones para reorganizar la entrada
        # Asumiendo 21 características como en el script anterior para la red MLP
        # Si las características de entrada cambian, este valor debe ajustarse.
        # Aquí lo configuramos para que el modelo funcione con 21 características
        # que es lo que se infiere del dataset diabetes.csv
        self.input_channels = 1  # Un canal para los datos
        self.input_height = input_size  # Altura de la matriz reorganizada (número de características)
        self.input_width = 1  # Ancho de la matriz reorganizada

        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(self.input_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        # Calcular el tamaño de salida después de las capas convolucionales
        # Para ello, necesitamos pasar un tensor de ejemplo a través de las capas convolucionales
        # Esto es un truco para calcular el tamaño de forma dinámica.
        # Creamos un tensor dummy para calcular el tamaño de salida.
        dummy_input = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        conv_output_dummy = self.conv_layers(dummy_input)
        conv_output_size = conv_output_dummy.view(conv_output_dummy.size(0), -1).size(1)

        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)  # Salida para 2 clases
        )

    def forward(self, x):
        # Reorganizar la entrada en una matriz 2D
        # x debería tener shape (batch_size, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)

        # Aplicar capas convolucionales
        x = self.conv_layers(x)

        # Aplanar para las capas fully connected
        x = x.view(batch_size, -1)

        # Aplicar capas fully connected
        x = self.fc_layers(x)

        return x


# --- Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(batch_size: int = 64):
    """
    Carga, filtra, preprocesa y balancea los datos del archivo CSV.
    Separa los datos en train, validation y test. Guarda los conjuntos de prueba y entrenamiento/validación en CSV.

    Args:
        batch_size (int): Tamaño del lote para el DataLoader.

    Returns:
        tuple: (train_loader, val_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor)
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Error: El archivo '{DATA_FILE}' no se encontró. Asegúrate de que esté en el mismo directorio que el script."
        )

    data = pd.read_csv(DATA_FILE)

    # Filtrar solo las clases 0 y 2 y luego mapear 2 a 1.
    data = data[data['Diabetes_012'].isin([0, 2])].copy()
    data['Diabetes_012'] = data['Diabetes_012'].replace({2: 1})

    print("\nVariables de entrada utilizadas:")
    input_variables = data.drop('Diabetes_012', axis=1).columns.tolist()
    for i, var in enumerate(input_variables, 1):
        print(f"{i}. {var}")

    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']

    print("\nDistribución de clases original (filtrada y binarizada):")
    print(Counter(y))

    # Primera división: separar el 20% de los datos para la prueba final
    # X_temp y y_temp serán usados para train y validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Guardar los conjuntos de prueba y entrenamiento/validación en CSV
    # El directorio ya debe haber sido creado por ensure_directories_exist en main()
    df_test_final = pd.concat([X_test, y_test], axis=1)
    test_csv_path = os.path.join(MODEL_SAVE_DIR, FINAL_TEST_CSV_NAME)
    df_test_final.to_csv(test_csv_path, index=False)
    print(f"\nConjunto de prueba final (20%) guardado en: {test_csv_path}")

    df_train_val = pd.concat([X_temp, y_temp], axis=1)
    train_val_csv_path = os.path.join(MODEL_SAVE_DIR, TRAIN_VAL_CSV_NAME)
    df_train_val.to_csv(train_val_csv_path, index=False)
    print(f"Conjunto de entrenamiento y validación (80%) guardado en: {train_val_csv_path}")

    # Segunda división: separar validation del conjunto temporal (ahora el 20% de los datos restantes)
    # 0.20 * (1 - 0.20) = 0.20 * 0.80 = 0.16 del total original
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(VAL_SIZE / (1 - TEST_SIZE)), random_state=SEED, stratify=y_temp
    )

    print("\nTamaños de los conjuntos:")
    print(f"Train: {len(X_train)} muestras")
    print(f"Validation: {len(X_val)} muestras")
    print(f"Test (20% final): {len(X_test)} muestras")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    input_size = X_train_scaled.shape[1]

    print("\nDistribución de clases en entrenamiento ANTES del balanceo:")
    print(Counter(y_train))

    # --- Cálculo de pesos de clase (manual o automático) ---
    if MANUAL_CLASS_WEIGHTS:
        class_weights_tensor = torch.FloatTensor(CLASS_WEIGHTS_VALUES)
        print(f"\nUsando pesos de clase manuales: {class_weights_tensor}")
    else:
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights_array)
        print(f"\nPesos de clase calculados automáticamente: {class_weights_tensor}")

    # SMOTE solo en el conjunto de entrenamiento (X_train_scaled, y_train)
    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print("\nDistribución de clases en entrenamiento DESPUÉS de SMOTE:")
    print(Counter(y_train_balanced))

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.FloatTensor(X_train_balanced)
    y_train_tensor = torch.LongTensor(y_train_balanced.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.values)

    # Crear DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=(device.type == 'cuda'))

    return train_loader, val_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor


# --- Funciones de Evaluación y Visualización ---
def evaluate_model(model: nn.Module, X_data: torch.Tensor, y_true: torch.Tensor, device: torch.device,
                   threshold: float = 0.5):
    """
    Evalúa el modelo en un conjunto de datos dado utilizando un umbral de decisión.

    Args:
        model (nn.Module): El modelo a evaluar.
        X_data (torch.Tensor): Tensores de características del conjunto de datos.
        y_true (torch.Tensor): Tensores de etiquetas reales del conjunto de datos.
        device (torch.device): El dispositivo (CPU/CUDA) donde está el modelo y los datos.
        threshold (float): Umbral de probabilidad para clasificar como clase 1 (Diabetes).

    Returns:
        tuple: (y_pred_np, accuracy, f1_score_diabetes, precision_diabetes, recall_diabetes, y_true_np)
               Retorna métricas clave y las predicciones/valores reales como arrays de numpy.
    """
    model.eval()
    with torch.no_grad():
        X_data_dev = X_data.to(device)
        outputs_dev = model(X_data_dev)

        # Aplicar Softmax para obtener probabilidades
        probabilities = torch.softmax(outputs_dev, dim=1)
        # La probabilidad de ser la clase 1 (Diabetes)
        prob_diabetes = probabilities[:, 1]

        # Clasificar basándose en el umbral
        predicted_dev = (prob_diabetes >= threshold).long()

        y_true_np = y_true.cpu().numpy()
        predicted_np = predicted_dev.cpu().numpy()

        accuracy = accuracy_score(y_true_np, predicted_np)
        precision_diabetes = precision_score(y_true_np, predicted_np, pos_label=1, zero_division=0)
        recall_diabetes = recall_score(y_true_np, predicted_np, pos_label=1, zero_division=0)
        f1_diabetes = f1_score(y_true_np, predicted_np, pos_label=1, zero_division=0)

    return predicted_np, accuracy, f1_diabetes, precision_diabetes, recall_diabetes, y_true_np


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_dir: str, file_name: str):
    """
    Genera y guarda una matriz de confusión en el directorio y con el nombre de archivo especificado.
    """
    try:
        # El directorio ya debería haber sido creado por ensure_directories_exist
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error al generar la matriz de confusión: {str(e)}")


def plot_metrics(train_losses, val_f1_scores_diabetes, val_accuracies, save_dir: str, file_name: str):
    """
    Genera y guarda gráficos de la pérdida de entrenamiento y métricas de validación/prueba.
    """
    try:
        # El directorio ya debería haber sido creado por ensure_directories_exist
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

        # Gráfico de Métricas de Validación (F1-Score Diabetes y Accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, val_f1_scores_diabetes, label='F1-Score (Diabetes) en Validación', color='orange')
        plt.plot(epochs_range, val_accuracies, label='Accuracy en Validación', color='green', linestyle='--')
        plt.title('Métricas de Validación por Época')
        plt.xlabel('Época')
        plt.ylabel('Valor de Métrica')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error al generar los gráficos de entrenamiento: {str(e)}")


# --- Función de Entrenamiento ---
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, class_weights: torch.Tensor = None,
                epochs: int = 100, lr: float = 0.001):
    """
    Entrena el modelo de red neuronal usando validación.

    Args:
        model (nn.Module): El modelo a entrenar.
        train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.
        val_loader (DataLoader): DataLoader para el conjunto de validación.
        device (torch.device): El dispositivo (CPU/CUDA) donde entrenar el modelo.
        class_weights (torch.Tensor, optional): Pesos de clase para CrossEntropyLoss.
        epochs (int): Número de épocas de entrenamiento.
        lr (float): Tasa de aprendizaje inicial.

    Returns:
        nn.Module: El modelo entrenado con el mejor rendimiento en validación.
        list: Lista de pérdidas de entrenamiento por época.
        list: Lista de F1-Scores en validación por época.
        list: Lista de Accuracies en validación por época.
        np.ndarray: Predicciones de validación de la última época.
        np.ndarray: Etiquetas reales de validación de la última época.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Usando pesos de clase en CrossEntropyLoss: {class_weights} en dispositivo {class_weights.device}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("No se usan pesos de clase en CrossEntropyLoss.")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, min_lr=1e-6)

    best_f1_diabetes = -1  # Se inicializa en -1 para asegurar que cualquier F1-score válido sea mejor
    best_model_state = None
    epochs_no_improve = 0
    early_stopping_patience = 25

    train_losses = []
    val_accuracies = []
    val_f1_scores_diabetes = []

    # Variables para guardar las predicciones y etiquetas de la última época de validación
    last_epoch_val_preds = np.array([])
    last_epoch_val_targets = np.array([])

    for epoch in range(epochs):
        # Fase de entrenamiento
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

        # Fase de validación
        model.eval()
        val_loss = 0.0
        current_val_preds = []
        current_val_targets = []

        with torch.no_grad():
            for data_batch, targets_batch in val_loader:
                data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)
                outputs = model(data_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item() * data_batch.size(0)

                probabilities = torch.softmax(outputs, dim=1)
                # Usar 0.5 como umbral para la validación interna
                preds = (probabilities[:, 1] >= 0.5).long()
                current_val_preds.extend(preds.cpu().numpy())
                current_val_targets.extend(targets_batch.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(current_val_targets, current_val_preds)
        val_f1 = f1_score(current_val_targets, current_val_preds, pos_label=1, zero_division=0)

        val_accuracies.append(val_accuracy)
        val_f1_scores_diabetes.append(val_f1)

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Val F1 (Diabetes): {val_f1:.4f}")

        # Guardar las predicciones y etiquetas de validación de la época actual
        # Solo necesitamos las de la última época, pero las actualizamos en cada iteración
        last_epoch_val_preds = np.array(current_val_preds)
        last_epoch_val_targets = np.array(current_val_targets)

        # Paso del scheduler basado en el F1-Score de validación
        scheduler.step(val_f1)

        # Lógica de Early Stopping basada en validación (F1-score de Diabetes)
        if val_f1 > best_f1_diabetes:
            best_f1_diabetes = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"→ Nuevo mejor modelo (Val F1-Score Diabetes): {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping activado después de {early_stopping_patience} épocas sin mejora en Val F1-Score.")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\nEntrenamiento finalizado. Mejor F1-Score (Diabetes) en validación: {best_f1_diabetes:.4f}")
    else:
        print("\nEntrenamiento finalizado. No se guardó ningún mejor modelo.")

    return model, train_losses, val_f1_scores_diabetes, val_accuracies, last_epoch_val_preds, last_epoch_val_targets


# --- Función Principal ---
def main():
    try:
        # Asegúrate de que los directorios existan al inicio
        ensure_directories_exist([MODEL_SAVE_DIR])

        train_loader, val_loader, X_test_tensor, y_test_tensor, input_size, class_weights_tensor = \
            load_and_preprocess_data(batch_size=BATCH_SIZE)

        model = DiabetesConvNet(input_size).to(device)  # Usa DiabetesConvNet

        print("\nIniciando entrenamiento del modelo convolucional...")
        trained_model, train_losses, val_f1_scores_diabetes, val_accuracies, \
            last_val_preds, last_val_targets = train_model(
            model, train_loader, val_loader, device,  # Se eliminó X_test_tensor, y_test_tensor de aquí
            class_weights=class_weights_tensor, epochs=EPOCHS, lr=LEARNING_RATE
        )

        # --- Guardar el modelo final ---
        model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"\nModelo final guardado en: {model_save_path}")

        # --- Generar y guardar gráficos de entrenamiento y validación ---
        print("\nGenerando gráficos de entrenamiento y validación...")
        plot_metrics(train_losses, val_f1_scores_diabetes, val_accuracies,
                     save_dir=MODEL_SAVE_DIR, file_name=TRAINING_METRICS_NAME)
        print(
            f"Gráfico de métricas de entrenamiento/validación guardado como: {os.path.join(MODEL_SAVE_DIR, TRAINING_METRICS_NAME)}")

        # --- Generar y guardar la matriz de confusión de la última época de validación ---
        print("\nGenerando Matriz de Confusión para la última época de validación...")
        plot_confusion_matrix(last_val_targets, last_val_preds, class_names=CLASS_NAMES,
                              save_dir=MODEL_SAVE_DIR, file_name=CONFUSION_MATRIX_NAME)
        print(
            f"Matriz de Confusión para la última época de validación guardada como: {os.path.join(MODEL_SAVE_DIR, CONFUSION_MATRIX_NAME)}")

        print(
            f"\nEvaluando el mejor modelo en el conjunto de prueba final (Umbral de decisión: {PREDICTION_THRESHOLD:.2f})...")
        y_pred_final, accuracy_final, f1_diabetes_final, precision_diabetes_final, recall_diabetes_final, y_true_final = \
            evaluate_model(trained_model, X_test_tensor, y_test_tensor, device, threshold=PREDICTION_THRESHOLD)

        print("\n--- Resultados Finales en el Conjunto de Prueba ---")

        # Guardar Classification Report en un archivo de texto
        report = classification_report(y_true_final, y_pred_final, target_names=CLASS_NAMES, zero_division=0)
        report_save_path = os.path.join(MODEL_SAVE_DIR, FINAL_CLASSIFICATION_REPORT_NAME)
        with open(report_save_path, 'w') as f:
            f.write(report)
        print(f"\nClassification Report guardado en: {report_save_path}")
        print("\nClassification Report (mostrado en consola):")
        print(report)

        print("\nMétricas detalladas por clase (sklearn.metrics):")
        print(f"Métricas para la clase 0 (No Diabetes):")
        print(
            f"Precisión: {precision_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")
        print(f"Recall: {recall_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")
        print(f"F1-Score: {f1_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")

        print(f"\nMétricas para la clase 1 (Diabetes):")
        print(f"Precisión: {precision_diabetes_final:.2%}")
        print(f"Recall: {recall_diabetes_final:.2%}")
        print(f"F1-Score: {f1_diabetes_final:.2%}")

        # --- Generar y guardar la matriz de confusión para el conjunto de prueba final ---
        print("\nGenerando Matriz de Confusión para el conjunto de prueba final...")
        plot_confusion_matrix(y_true_final, y_pred_final, class_names=CLASS_NAMES,
                              save_dir=MODEL_SAVE_DIR, file_name=FINAL_CONFUSION_MATRIX_NAME)
        print(
            f"Matriz de Confusión para prueba final guardada como: {os.path.join(MODEL_SAVE_DIR, FINAL_CONFUSION_MATRIX_NAME)}")

        print("\nProceso finalizado.")

    except FileNotFoundError as e:
        print(e)
        print("Asegúrate de que el archivo 'diabetes.csv' esté en el directorio correcto.")
    except Exception as e:
        print(f"Error fatal en la ejecución principal: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()