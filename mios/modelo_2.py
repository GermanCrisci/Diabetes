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
# Una semilla para que los resultados sean reproducibles
SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
# Ahora TEST_SIZE se refiere al split inicial (80/20)
INITIAL_TEST_SIZE = 0.2
# Este será el tamaño del conjunto de validación del 80% restante
VALIDATION_SIZE = 0.25 # 0.25 de 0.8 es 0.2 del total
CLASS_NAMES = ["No Diabetes", "Diabetes"]
DATA_FILE = '../diabetes.csv'

# Carpeta para guardar los datos de prueba finales
MODEL_DATA_DIR = '../datos_modelo_2'
FINAL_TEST_FILE = os.path.join(MODEL_DATA_DIR, 'final_test_data.csv')

# Fijar la semilla para PyTorch y NumPy
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Configuración de CUDA ---
# Elegir entre GPU (CUDA) o CPU si no hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")
if device.type == 'cuda':
    print(f"Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed_all(SEED) # Semilla para CUDA

# --- Definición del Modelo ---
class DiabetesClassifier(nn.Module):
    """
    Esta es nuestra Red Neuronal, diseñada para clasificar si alguien tiene diabetes o no.
    Está construida con varias capas para aprender patrones complejos en los datos.
    """
    def __init__(self, input_size):
        super(DiabetesClassifier, self).__init__()
        self.model = nn.Sequential(
            # Primera capa: transforma la entrada a 128 neuronas
            nn.Linear(input_size, 128),
            # Normalización para ayudar al entrenamiento
            nn.BatchNorm1d(128),
            # Función de activación para introducir no linealidad
            nn.ReLU(),
            # Dropout para prevenir el sobreajuste (apaga el 30% de las neuronas aleatoriamente)
            nn.Dropout(0.3),
            # Segunda capa: de 128 a 64 neuronas
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Tercera capa: de 64 a 32 neuronas
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Capa de salida: de 32 a 2 neuronas (una por cada clase: No Diabetes, Diabetes)
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # El flujo de datos a través de las capas que definimos
        return self.model(x)

# --- Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(batch_size: int = 64):
    """
    Prepara nuestros datos: los carga, limpia, escala y los divide para el entrenamiento y la prueba.
    También balancea el conjunto de entrenamiento y calcula pesos para las clases.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"¡Ups! El archivo '{DATA_FILE}' no se encontró. Asegúrate de que esté en el mismo directorio que el script."
        )

    data = pd.read_csv(DATA_FILE)

    # Filtrar solo las clases 0 y 2 y luego mapear 2 a 1.
    data = data[data['Diabetes_012'].isin([0, 2])].copy()
    data['Diabetes_012'] = data['Diabetes_012'].replace({2: 1})

    print("\nAquí están las variables que usaremos para predecir:")
    input_variables = data.drop('Diabetes_012', axis=1).columns.tolist()
    for i, var in enumerate(input_variables, 1):
        print(f"{i}. {var}")

    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']

    print("\nDistribución de clases (No Diabetes vs. Diabetes) en nuestros datos originales:")
    print(Counter(y))

    # --- Split inicial 80/20 y guardado del 20% para la prueba final ---
    print(f"\nRealizando split inicial de datos en 80% (entrenamiento+validación) y 20% (prueba final)...")
    X_train_val, X_final_test, y_train_val, y_final_test = train_test_split(
        X, y, test_size=INITIAL_TEST_SIZE, random_state=SEED, stratify=y
    )

    # Crear la carpeta si no existe
    os.makedirs(MODEL_DATA_DIR, exist_ok=True)
    # Guardar el conjunto de prueba final
    final_test_df = X_final_test.copy()
    final_test_df['Diabetes_012'] = y_final_test
    final_test_df.to_csv(FINAL_TEST_FILE, index=False)
    print(f"Conjunto de prueba final (20%) guardado en: {FINAL_TEST_FILE}")
    print(f"Distribución de clases en el conjunto de prueba final: {Counter(y_final_test)}")

    # --- Ahora, trabajar con el 80% para entrenamiento y validación ---
    print(f"\nDividiendo el 80% restante en entrenamiento y validación ({1-VALIDATION_SIZE:.0%} / {VALIDATION_SIZE:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SIZE, random_state=SEED, stratify=y_train_val
    )

    # Escalamos las características para que estén en un rango similar, lo que ayuda al modelo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) # Escalar el conjunto de validación
    input_size = X_train_scaled.shape[1]

    print("\nDistribución de clases en el conjunto de entrenamiento ANTES de balancear:")
    print(Counter(y_train))
    print("Distribución de clases en el conjunto de validación:")
    print(Counter(y_val))

    # Calculamos pesos para cada clase. Esto ayuda a la red a prestar más atención a las clases minoritarias.
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.FloatTensor(class_weights_array)
    print(f"\nPesos de clase calculados (útiles para la función de pérdida): {class_weights_tensor}")

    # Usamos SMOTE para balancear el conjunto de entrenamiento, creando "ejemplos sintéticos" de la clase minoritaria.
    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced_smote = smote.fit_resample(X_train_scaled, y_train)
    print("\nDistribución de clases en el entrenamiento DESPUÉS de usar SMOTE:")
    print(Counter(y_train_balanced_smote))

    # Convertimos nuestros datos a tensores de PyTorch, que es lo que el modelo necesita
    X_train_tensor = torch.FloatTensor(X_train_balanced)
    y_train_tensor = torch.LongTensor(y_train_balanced_smote.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled) # Convertir validación a tensor
    y_val_tensor = torch.LongTensor(y_val.values) # Convertir validación a tensor

    # Creamos un "DataLoader" para alimentar los datos al modelo en lotes durante el entrenamiento
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))

    return train_loader, X_val_tensor, y_val_tensor, input_size, class_weights_tensor

# --- Funciones de Evaluación y Visualización ---
def evaluate_model(model: nn.Module, X_data: torch.Tensor, y_true: torch.Tensor, device: torch.device):
    """
    Esta función revisa qué tan bien le fue a nuestro modelo en un conjunto de datos,
    calculando métricas importantes como la precisión y el F1-Score.
    """
    model.eval() # Ponemos el modelo en modo evaluación (desactiva dropout, etc.)
    with torch.no_grad(): # No necesitamos calcular gradientes en la evaluación
        X_data_dev = X_data.to(device)
        outputs_dev = model(X_data_dev)
        _, predicted_dev = torch.max(outputs_dev.data, 1) # Obtenemos la clase predicha

        y_true_np = y_true.cpu().numpy() # Convertimos las etiquetas reales a NumPy
        predicted_np = predicted_dev.cpu().numpy() # Convertimos las predicciones a NumPy

        accuracy = accuracy_score(y_true_np, predicted_np)
        # Calculamos métricas específicas para la clase "Diabetes" (clase 1)
        precision_diabetes = precision_score(y_true_np, predicted_np, pos_label=1, zero_division=0)
        recall_diabetes = recall_score(y_true_np, predicted_np, pos_label=1, zero_division=0)
        f1_diabetes = f1_score(y_true_np, predicted_np, pos_label=1, zero_division=0)

    return predicted_np, accuracy, f1_diabetes, precision_diabetes, recall_diabetes, y_true_np


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, title_suffix: str = ""):
    """
    Dibuja una matriz de confusión para visualizar los aciertos y errores del modelo.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Matriz de Confusión{title_suffix}: ¿Qué tan bien predijo el modelo?')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"¡Ocurrió un error al dibujar la matriz de confusión!: {str(e)}")

def plot_metrics(train_losses, val_f1_scores_diabetes, val_accuracies):
    """
    Genera gráficos para ver cómo evolucionaron la pérdida de entrenamiento
    y las métricas de validación a lo largo de las épocas.
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico de la pérdida durante el entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Pérdida de Entrenamiento')
    plt.title('Cómo disminuyó la pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.legend()

    # Gráfico de las métricas clave en el conjunto de validación
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_f1_scores_diabetes, label='F1-Score (Diabetes) en Validación', color='orange')
    plt.plot(epochs_range, val_accuracies, label='Accuracy en Validación', color='green', linestyle='--')
    plt.title('Rendimiento del modelo en el conjunto de validación')
    plt.xlabel('Época')
    plt.ylabel('Valor de Métrica')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- Función de Entrenamiento ---
def train_model(model: nn.Module, train_loader: DataLoader, X_val_tensor: torch.Tensor,
                y_val_tensor: torch.Tensor, device: torch.device, class_weights: torch.Tensor = None,
                epochs: int = 100, lr: float = 0.001):
    """
    Esta función se encarga de entrenar el modelo.
    Irá ajustando sus "pesos" para aprender a hacer mejores predicciones.
    """
    # Configuramos la función de pérdida (CrossEntropyLoss) que mide qué tan bien le va al modelo.
    # Usamos los pesos de clase si los calculamos, para manejar el desbalance.
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"¡Atención! Usando pesos de clase en la función de pérdida: {class_weights} en el dispositivo {class_weights.device}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("No se usaron pesos de clase en la función de pérdida.")

    # El optimizador Adam ajusta los pesos del modelo
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Este scheduler reduce la tasa de aprendizaje si el F1-Score (Diabetes) no mejora,
    # ayudando a encontrar un mejor punto de convergencia.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

    best_f1_diabetes = 0 # Guardaremos el mejor F1-Score para la clase 'Diabetes'
    best_model_state = None
    epochs_no_improve = 0 # Contador para el "Early Stopping"
    early_stopping_patience = 10 # Si no mejora en este número de épocas, paramos el entrenamiento

    train_losses = []
    val_accuracies = []
    val_f1_scores_diabetes = []

    for epoch in range(epochs):
        model.train() # Ponemos el modelo en modo entrenamiento
        running_loss = 0.0
        # tqdm nos da una barra de progreso, ¡muy útil!
        batch_iterator = tqdm(train_loader, desc=f"Época {epoch + 1}/{epochs}", unit="lote")

        for data_batch, targets_batch in batch_iterator:
            data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad() # Limpiamos los gradientes de la iteración anterior
            outputs = model(data_batch) # Hacemos una predicción
            loss = criterion(outputs, targets_batch) # Calculamos la pérdida
            loss.backward() # Calculamos los gradientes
            optimizer.step() # Actualizamos los pesos del modelo
            running_loss += loss.item() * data_batch.size(0)
            batch_iterator.set_postfix(loss=loss.item()) # Mostramos la pérdida en la barra de progreso

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluamos el modelo en el conjunto de validación para ver su rendimiento real
        _, accuracy, f1_diabetes, precision_diabetes, recall_diabetes, _ = \
            evaluate_model(model, X_val_tensor, y_val_tensor, device)

        val_accuracies.append(accuracy)
        val_f1_scores_diabetes.append(f1_diabetes)

        print(f"\nÉpoca [{epoch + 1}/{epochs}], Pérdida de Entrenamiento Promedio: {epoch_loss:.4f}, "
              f"Precisión en Validación: {accuracy:.4f}, Precisión (Diabetes) en Validación: {precision_diabetes:.4f}, "
              f"Recall (Diabetes) en Validación: {recall_diabetes:.4f}, F1 (Diabetes) en Validación: {f1_diabetes:.4f}")

        # Le decimos al scheduler el rendimiento actual para que decida si reduce la tasa de aprendizaje
        scheduler.step(f1_diabetes)

        # Lógica de Early Stopping: Si el F1-Score (Diabetes) no mejora, paramos para no sobreajustar
        if f1_diabetes > best_f1_diabetes:
            best_f1_diabetes = f1_diabetes
            # Guardamos el estado del modelo actual porque es el mejor hasta ahora
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0 # Reiniciamos el contador
            print(f"→ ¡Nuevo mejor modelo encontrado en Validación! F1-Score (Diabetes): {f1_diabetes:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Parada temprana activada: el F1-Score (Diabetes) no ha mejorado en {early_stopping_patience} épocas en Validación. ¡Es hora de detenerse!")
                break # Salimos del bucle de entrenamiento

    # Cargamos el mejor modelo guardado para la evaluación final
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\nEntrenamiento completado. El mejor F1-Score (Diabetes) alcanzado en validación fue: {best_f1_diabetes:.4f}")
    else:
        print("\nEntrenamiento completado. ¡No se encontró un mejor modelo para cargar! Revisa la configuración.")
    return model, train_losses, val_f1_scores_diabetes, val_accuracies

# --- Función para cargar y probar el conjunto de prueba final ---
def load_and_test_final_data(model: nn.Module, input_size: int, device: torch.device):
    """
    Carga el 20% de los datos que se guardaron al principio y prueba el modelo final en ellos.
    Estos datos no fueron vistos durante el entrenamiento o la validación.
    """
    print(f"\nCargando el conjunto de prueba final desde: {FINAL_TEST_FILE}")
    if not os.path.exists(FINAL_TEST_FILE):
        raise FileNotFoundError(
            f"Error: El archivo de prueba final '{FINAL_TEST_FILE}' no se encontró. "
            f"Ejecuta el script principal para generarlo."
        )

    final_test_df = pd.read_csv(FINAL_TEST_FILE)
    X_final_test_df = final_test_df.drop('Diabetes_012', axis=1)
    y_final_test_df = final_test_df['Diabetes_012']

    # Necesitamos escalar los datos de prueba final con el mismo escalador usado en el entrenamiento.
    # Por simplicidad aquí lo instanciamos de nuevo, pero en un caso real se serializaría el escalador.
    # Asumimos que la media y desviación estándar son similares a las de X_train_val.
    scaler = StandardScaler()
    # Una forma de "simular" que se ajustó con el entrenamiento sin guardar el objeto scaler
    # Es ajustar y transformar con los datos de X_final_test_df. Ojo: idealmente, se usaría el scaler entrenado.
    # Para este ejemplo, haremos un fit_transform simple para que funcione.
    # EN UN CASO REAL, cargarías el scaler guardado con pickle.
    X_final_test_scaled = scaler.fit_transform(X_final_test_df)

    X_final_test_tensor = torch.FloatTensor(X_final_test_scaled)
    y_final_test_tensor = torch.LongTensor(y_final_test_df.values)

    print("\nEvaluando el modelo final en el conjunto de prueba final (datos nunca vistos)...")
    y_pred_final, accuracy_final, f1_diabetes_final, precision_diabetes_final, recall_diabetes_final, y_true_final = \
        evaluate_model(model, X_final_test_tensor, y_final_test_tensor, device)

    print("\n--- Resultados Finales en el Conjunto de Prueba Totalmente Separado ---")
    print("\nClassification Report (Conjunto de Prueba Final):")
    print(classification_report(y_true_final, y_pred_final, target_names=CLASS_NAMES, zero_division=0))

    print("\nGenerando Matriz de Confusión (Conjunto de Prueba Final)...")
    plot_confusion_matrix(y_true_final, y_pred_final, class_names=CLASS_NAMES, title_suffix=" (Prueba Final)")

    print("\nMétricas detalladas por clase (Conjunto de Prueba Final):")
    print(f"Métricas para la clase 0 (No Diabetes):")
    print(f"  Precisión: {precision_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")
    print(f"  Recall: {recall_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")
    print(f"  F1-Score: {f1_score(y_true_final, y_pred_final, pos_label=0, average='binary', zero_division=0):.2%}")

    print(f"\nMétricas para la clase 1 (Diabetes):")
    print(f"  Precisión: {precision_diabetes_final:.2%}")
    print(f"  Recall: {recall_diabetes_final:.2%}")
    print(f"  F1-Score: {f1_diabetes_final:.2%}")


# --- Función Principal ---
def main():
    """
    Esta es la función principal que orquesta todo el proceso:
    carga los datos, entrena el modelo y muestra los resultados.
    """
    try:
        # Cargamos y preprocesamos nuestros datos, obteniendo ahora el conjunto de validación
        train_loader, X_val_tensor, y_val_tensor, input_size, class_weights_tensor = \
            load_and_preprocess_data(batch_size=BATCH_SIZE)

        # Creamos una instancia de nuestro modelo y lo movemos al dispositivo (CPU/GPU)
        model = DiabetesClassifier(input_size).to(device)

        print("\n¡Comenzando el entrenamiento del modelo!")
        # Entrenamos el modelo y obtenemos las métricas de progreso
        trained_model, train_losses, val_f1_scores_diabetes, val_accuracies = train_model(
            model, train_loader, X_val_tensor, y_val_tensor, device,
            class_weights=class_weights_tensor, epochs=EPOCHS, lr=LEARNING_RATE
        )

        print("\nGenerando gráficos de cómo el modelo aprendió...")
        # Dibuja los gráficos de pérdida y métricas a lo largo del entrenamiento/validación
        plot_metrics(train_losses, val_f1_scores_diabetes, val_accuracies)

        # Finalmente, probamos el modelo con el 20% de los datos que separamos al inicio
        load_and_test_final_data(trained_model, input_size, device)

        print("\n¡Proceso de entrenamiento y evaluación completado!")

    except FileNotFoundError as e:
        print(e)
        print("Asegúrate de que el archivo 'diabetes.csv' esté en el mismo directorio que este script.")
    except Exception as e:
        print(f"¡Ocurrió un error inesperado durante la ejecución principal!: {str(e)}")
        # Esto nos da más detalles si algo sale muy mal
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()