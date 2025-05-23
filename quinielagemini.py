import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
from collections import Counter

# Configuración global para pandas y advertencias
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

class QuinielaPredictor:
    """
    Clase para predecir números de quiniela utilizando datos históricos y diversas estrategias.
    Permite cargar datos, entrenar un modelo predictivo y generar números basados en
    el modelo, números calientes, fríos, una mezcla balanceada o aleatoriedad pura.
    """
    def __init__(self):
        self.model = None # Modelo de Machine Learning (RandomForestClassifier)
        self.model_trained = False # Bandera para saber si el modelo ha sido entrenado
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.ultimo_dia = None # El último día con datos en el archivo histórico
        self.test_size = 0.2 # Proporción de datos para el conjunto de prueba
        self.random_state = None # Semilla aleatoria para el modelo (None para verdadera aleatoriedad)
        self.all_possible_numbers = list(range(0, 37)) # Rango de números posibles en la quiniela (0 a 36)
        self.hot_numbers = [] # Números que han salido con mayor frecuencia recientemente
        self.cold_numbers = [] # Números que han salido con menor frecuencia o han estado ausentes
        self.most_frequent_overall = [] # Números más frecuentes en todo el historial

    def load_data(self, filepath):
        """
        Carga datos históricos de la quiniela desde un archivo Excel.
        Detecta el último día con datos y asegura que los números estén en el rango 0-36,
        tratando el 0 como un número válido.
        
        Args:
            filepath (str): Ruta al archivo Excel con los datos.
            
        Returns:
            pd.DataFrame: DataFrame con los datos limpios y cargados.
        """
        try:
            # Cargar el archivo Excel, asumiendo que no tiene encabezado y usa 8 columnas
            df = pd.read_excel(filepath, header=None, usecols=range(8))
            df.columns = ['hora'] + self.dias_semana # Asignar nombres de columna

            # Detectar el último día con datos en el DataFrame
            for dia in reversed(self.dias_semana):
                if not df[dia].isnull().all(): # Si la columna no está completamente vacía
                    self.ultimo_dia = dia
                    break

            if self.ultimo_dia is None:
                raise ValueError("El archivo no contiene datos válidos en ninguna columna de día.")

            # Limpieza de datos: convertir a numérico y filtrar por el rango 0-36
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Convertir a numérico, errores a NaN
                # Filtrar solo números dentro del rango 0-36 (incluyendo el 0)
                df[col] = df[col].apply(lambda x: x if 0 <= x <= 36 else np.nan)

            return df.dropna() # Eliminar filas con cualquier valor NaN después de la limpieza
        except FileNotFoundError:
            print(f"Error: El archivo '{filepath}' no se encontró.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error cargando datos desde '{filepath}': {str(e)}")
            return pd.DataFrame()

    def get_next_day(self):
        """
        Calcula el próximo día de la semana basándose en el último día con datos.
        
        Returns:
            str: El nombre del próximo día de la semana en minúsculas.
        """
        if self.ultimo_dia is None:
            raise ValueError("Primero debe cargar los datos históricos para determinar el último día.")

        current_idx = self.dias_semana.index(self.ultimo_dia)
        next_idx = (current_idx + 1) % 7 # Asegura que el índice se reinicie a 0 después de domingo
        return self.dias_semana[next_idx]

    def transform_to_long_format(self, df):
        """
        Transforma el DataFrame de formato ancho a formato largo.
        Cada fila representa una ocurrencia de un número en un día y hora específicos.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos en formato ancho.
            
        Returns:
            pd.DataFrame: DataFrame con los datos en formato largo.
        """
        records = []
        for _, row in df.iterrows():
            for dia in self.dias_semana:
                num = row[dia]
                if pd.notnull(num): # Solo procesar números válidos (no NaN)
                    records.append({
                        'dia_semana': dia,
                        'hora': int(row['hora']),
                        'numero': int(num),
                        'dia_num': self.dias_semana.index(dia), # Representación numérica del día
                        'es_finde': int(dia in ['sábado', 'domingo']) # 1 si es fin de semana, 0 si no
                    })
        return pd.DataFrame(records)

    def calculate_number_stats(self, df_long):
        """
        Calcula y almacena estadísticas de números:
        - Números más frecuentes en todo el historial.
        - Números "calientes" (más frecuentes en datos recientes).
        - Números "fríos" (menos frecuentes en datos recientes o ausentes).
        
        Args:
            df_long (pd.DataFrame): DataFrame con los datos en formato largo.
        """
        if df_long.empty:
            print("No hay datos para calcular estadísticas de números.")
            return

        # 1. Frecuencia general (para 'most_frequent_overall')
        num_counts_overall = Counter(df_long['numero'])
        # Tomar los 8 números más comunes en todo el historial
        self.most_frequent_overall = [num for num, _ in num_counts_overall.most_common(8)]

        # 2. Frecuencia reciente (para 'hot_numbers' y 'cold_numbers')
        # Considerar los últimos 100 registros para definir "reciente" (ajustable según volumen de datos)
        recent_data = df_long.tail(min(len(df_long), 100))
        recent_counts = Counter(recent_data['numero'])

        # Números calientes: top 5 de los más frecuentes en los datos recientes
        self.hot_numbers = [num for num, _ in recent_counts.most_common(5)]

        # Números fríos: aquellos que no aparecen en los datos recientes o son muy poco frecuentes
        all_numbers_set = set(self.all_possible_numbers)
        recent_numbers_set = set(recent_data['numero'].unique())

        # Números que no han aparecido en los datos recientes
        not_in_recent = list(all_numbers_set - recent_numbers_set)

        # Para los números que sí aparecen en los datos recientes, encontrar los menos frecuentes
        least_frequent_in_recent = sorted(recent_counts.items(), key=lambda item: item[1])
        # Combinar los que no están en reciente con los menos frecuentes de reciente
        self.cold_numbers = not_in_recent + [num for num, _ in least_frequent_in_recent[:5] if num not in not_in_recent]
        self.cold_numbers = list(set(self.cold_numbers)) # Eliminar duplicados si los hubiera

        print(f"\n🔢 Números más frecuentes (histórico): {self.most_frequent_overall}")
        print(f"🔥 Números calientes (recientes): {self.hot_numbers}")
        print(f"❄️ Números fríos (menos recientes/ausentes): {self.cold_numbers}")

    def create_features(self, df):
        """
        Prepara características adicionales para el entrenamiento del modelo.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos en formato largo.
            
        Returns:
            pd.DataFrame: DataFrame con las nuevas características añadidas.
        """
        df['hora_dia'] = df['hora'] * (df['dia_num'] + 1) # Característica combinada de hora y día
        df['es_par'] = (df['numero'] % 2 == 0).astype(int) # 1 si es par, 0 si es impar
        df['es_bajo'] = (df['numero'] <= 18).astype(int) # 1 si es bajo (0-18), 0 si es alto (19-36)
        return df

    def train_model(self, df):
        """
        Entrena el modelo de RandomForestClassifier con los datos históricos.
        Incluye validación y un respaldo a un modelo probabilístico si los datos son insuficientes.
        
        Args:
            df (pd.DataFrame): DataFrame con las características y el número objetivo.
            
        Returns:
            object: El modelo entrenado o la cadena "probabilistico" si se usa el modelo de respaldo.
        """
        try:
            # Si hay pocos datos, se usa un modelo probabilístico simple
            if len(df) < 50: # Se aumentó el umbral para un entrenamiento más significativo
                print("\n⚠️ Pocos datos para entrenar un modelo robusto. Usando modelo probabilístico.")
                return self.train_probabilistic_model(df)

            # Características (X) y variable objetivo (y)
            X = df[['hora', 'dia_num', 'hora_dia', 'es_finde', 'es_par', 'es_bajo']]
            y = df['numero']

            # División de datos en conjuntos de entrenamiento y prueba
            # Se evita 'stratify' si hay muchas clases únicas o clases con muy pocos ejemplos
            # para prevenir errores en la división.
            if y.nunique() > 20 and (y.value_counts() < 2).any():
                 X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.test_size,
                    random_state=42 # Semilla fija para reproducibilidad de la división
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.test_size,
                    stratify=y, # Intenta mantener la proporción de clases en los conjuntos
                    random_state=42 # Semilla fija para reproducibilidad de la división
                )

            print(f"\n📊 División de datos: {len(X_train)} registros de entrenamiento, {len(X_test)} de prueba.")

            # Inicialización y entrenamiento del RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200, # Aumentado el número de árboles para un mejor rendimiento
                random_state=self.random_state, # None para aleatoriedad real en cada ejecución
                class_weight='balanced' # Ayuda a manejar clases desequilibradas (números que aparecen más)
            )
            self.model.fit(X_train, y_train)
            self.model_trained = True

            print("\n📈 Evaluación del modelo:")
            self.evaluate_model(X_test, y_test) # Evaluar el modelo entrenado

            return self.model
        except Exception as e:
            print(f"\n❌ Error entrenando modelo de RandomForest: {str(e)}")
            print("Intentando con modelo probabilístico como respaldo.")
            return self.train_probabilistic_model(df) # Fallback si hay un error

    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el rendimiento del modelo entrenado utilizando métricas de clasificación.
        
        Args:
            X_test (pd.DataFrame): Características del conjunto de prueba.
            y_test (pd.Series): Variable objetivo del conjunto de prueba.
        """
        if not self.model_trained or self.model == "probabilistico":
            print("Modelo no entrenado o es probabilístico, no se puede evaluar con métricas de clasificación avanzadas.")
            return

        y_pred = self.model.predict(X_test) # Realizar predicciones en el conjunto de prueba

        print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
        # El reporte de clasificación puede ser muy extenso para muchos números,
        # se comenta para no abrumar la salida, pero se puede descomentar para depuración.
        # print("\nReporte de clasificación:")
        # print(classification_report(y_test, y_pred, zero_division=0))

    def train_probabilistic_model(self, df):
        """
        Entrena un modelo probabilístico simple basado en la frecuencia de aparición
        de los números para cada combinación de hora y día.
        Este es el modelo de respaldo cuando no hay suficientes datos para RandomForest.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos en formato largo.
            
        Returns:
            str: La cadena "probabilistico" para indicar que se usó este modelo.
        """
        self.model = "probabilistico"
        self.model_trained = True
        # Calcular las probabilidades de cada número para cada combinación de hora y día
        self.probabilidades = df.groupby(['hora', 'dia_num'])['numero'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()
        print("Modelo probabilístico entrenado (basado en frecuencias).")
        return self.model

    def predict_day(self, dia_semana, strategy="modelo"):
        """
        Genera una lista de números de quiniela para un día específico
        utilizando la estrategia de predicción seleccionada.
        
        Args:
            dia_semana (str): El día de la semana para el que se quiere predecir (ej. 'lunes').
            strategy (str): La estrategia a usar ('modelo', 'calientes', 'frios', 'balanceado', 'aleatorio').
            
        Returns:
            pd.DataFrame: DataFrame con la hora y los números predichos para el día.
        """
        try:
            dia_num = self.dias_semana.index(dia_semana.lower())
        except ValueError:
            raise ValueError(f"Día debe ser uno de: {', '.join(self.dias_semana)}")

        horas = [10, 11, 12, 13, 14, 15] # Horas para las que se generan predicciones
        predicciones = []
        
        print(f"\n✨ Generando números con estrategia: {strategy.capitalize()}...")

        for hora in horas:
            predicted_num = None
            
            # Estrategia: Modelo Predictivo (RandomForest)
            if strategy == "modelo" and self.model_trained and self.model != "probabilistico":
                # Crear un DataFrame con las características para la nueva predicción
                X_new = pd.DataFrame([{
                    'hora': hora,
                    'dia_num': dia_num,
                    'hora_dia': hora * (dia_num + 1),
                    'es_finde': int(dia_num >= 5),
                    # Para predecir, las características 'es_par' y 'es_bajo' se deben simular
                    # o basarse en una suposición, ya que no conocemos el número real de antemano.
                    # Aquí se usan valores aleatorios para introducir variedad.
                    'es_par': np.random.choice([0, 1]), 
                    'es_bajo': np.random.choice([0, 1])
                }])
                try:
                    # Predecir probabilidades para cada número y luego muestrear uno basado en esas probabilidades
                    proba = self.model.predict_proba(X_new)[0]
                    numbers = self.model.classes_
                    
                    # Asegurarse de que solo se consideren números dentro del rango 0-36
                    valid_indices = [i for i, num in enumerate(numbers) if num in self.all_possible_numbers]
                    if valid_indices:
                        filtered_numbers = [numbers[i] for i in valid_indices]
                        filtered_proba = [proba[i] for i in valid_indices]
                        
                        # Normalizar las probabilidades para que sumen 1
                        sum_proba = sum(filtered_proba)
                        if sum_proba > 0:
                            normalized_proba = [p / sum_proba for p in filtered_proba]
                            predicted_num = np.random.choice(filtered_numbers, p=normalized_proba)
                        else:
                            # Fallback si las probabilidades son cero o no válidas
                            predicted_num = np.random.choice(self.all_possible_numbers)
                    else:
                        predicted_num = np.random.choice(self.all_possible_numbers)
                except Exception as e:
                    print(f"Error en predicción del modelo para hora {hora}: {e}. Usando número aleatorio de respaldo.")
                    predicted_num = np.random.choice(self.all_possible_numbers)
            
            # Estrategia: Modelo Probabilístico (si el modelo principal no se entrenó)
            elif strategy == "probabilistico" and self.model == "probabilistico":
                if (hora, dia_num) in self.probabilidades:
                    nums = list(self.probabilidades[(hora, dia_num)].keys())
                    probs = list(self.probabilidades[(hora, dia_num)].values())
                    predicted_num = np.random.choice(nums, p=probs)
                else:
                    # Fallback a números frecuentes si no hay datos específicos para esa hora/día
                    predicted_num = np.random.choice(self.most_frequent_overall)

            # Estrategia: Números Calientes
            elif strategy == "calientes":
                predicted_num = np.random.choice(self.hot_numbers if self.hot_numbers else self.most_frequent_overall)
            
            # Estrategia: Números Fríos
            elif strategy == "frios":
                predicted_num = np.random.choice(self.cold_numbers if self.cold_numbers else self.most_frequent_overall)
            
            # Estrategia: Balanceado (mezcla de calientes, fríos y frecuentes)
            elif strategy == "balanceado":
                combined_numbers = list(set(self.hot_numbers + self.cold_numbers + self.most_frequent_overall))
                if not combined_numbers: # Si la lista combinada está vacía, usar todos los números posibles
                    combined_numbers = self.all_possible_numbers
                predicted_num = np.random.choice(combined_numbers)
            
            # Estrategia: Aleatorio Puro (por defecto o si la estrategia es inválida)
            else: 
                predicted_num = np.random.choice(self.all_possible_numbers)
            
            predicciones.append(f"{predicted_num:02d}") # Formatear a dos dígitos (ej. 05 en lugar de 5)

        # Heurística para asegurar variedad en las predicciones si son demasiado similares
        # Si menos de la mitad de las predicciones son únicas, se intenta añadir más variedad
        if len(set(predicciones)) < len(horas) / 2 and len(self.all_possible_numbers) >= len(horas):
            unique_preds = list(set(predicciones))
            remaining_needed = len(horas) - len(unique_preds)
            if remaining_needed > 0:
                # Elegir números adicionales que no se hayan predicho ya
                additional_numbers = np.random.choice(
                    list(set(self.all_possible_numbers) - set(int(p) for p in unique_preds)),
                    size=min(remaining_needed, len(self.all_possible_numbers) - len(unique_preds)), # Asegurar que no se pidan más números de los disponibles
                    replace=False # No repetir números adicionales
                )
                predicciones = unique_preds + [f"{num:02d}" for num in additional_numbers]
                np.random.shuffle(predicciones) # Mezclar para que no haya un patrón obvio

        return pd.DataFrame({
            'Hora': [f"{h}:00" for h in horas],
            'Predicción': predicciones,
            'Día': dia_semana.capitalize()
        })

    def fallback_predict(self, dia_semana):
        """
        Predicción de respaldo que utiliza los números más frecuentes en general
        cuando el modelo principal no está entrenado o falla.
        
        Args:
            dia_semana (str): El día de la semana para el que se quiere predecir.
            
        Returns:
            pd.DataFrame: DataFrame con la hora y los números predichos.
        """
        horas = [10, 11, 12, 13, 14, 15]
        # Usar los números más frecuentes en general, o todos los posibles si no hay frecuentes
        numbers_to_use = self.most_frequent_overall if self.most_frequent_overall else self.all_possible_numbers
        
        # Generar predicciones aleatorias a partir de los números seleccionados
        predicciones = [f"{num:02d}" for num in np.random.choice(numbers_to_use, size=len(horas), replace=True)]
        
        return pd.DataFrame({
            'Hora': [f"{h}:00" for h in horas],
            'Predicción': predicciones,
            'Día': dia_semana.capitalize()
        })

def get_user_choice(siguiente_dia, predictor_instance):
    """
    Obtiene la elección del usuario para el día a predecir y la estrategia de generación de números.
    
    Args:
        siguiente_dia (str): El día siguiente sugerido por el script.
        predictor_instance (QuinielaPredictor): Instancia del predictor para acceder a los días de la semana.
        
    Returns:
        tuple: Una tupla que contiene (dia_a_predecir, strategy).
    """
    # Selección del día a predecir
    while True:
        print("\n--- Opciones de Día ---")
        print(f"1. Predecir el siguiente día ({siguiente_dia.capitalize()})")
        print("2. Elegir otro día específico")
        
        opcion_dia = input("Seleccione una opción (1/2): ").strip()
        
        if opcion_dia == "1":
            dia_a_predecir = siguiente_dia
            break
        elif opcion_dia == "2":
            while True:
                dia_input = input(f"Ingrese el día a predecir ({', '.join(predictor_instance.dias_semana)}): ").lower()
                if dia_input in predictor_instance.dias_semana:
                    dia_a_predecir = dia_input
                    break
                print(f"❌ Error: Día debe ser uno de: {', '.join(predictor_instance.dias_semana)}")
            break
        else:
            print("❌ Error: Opción de día debe ser 1 o 2")

    # Selección de la estrategia de generación de números
    while True:
        print("\n--- Estrategias de Generación de Números ---")
        print("a. Modelo Predictivo (basado en patrones históricos)")
        print("b. Números Calientes (más frecuentes recientemente)")
        print("c. Números Fríos (menos frecuentes recientemente/ausentes)")
        print("d. Balanceado (mezcla de calientes, fríos y frecuentes)")
        print("e. Aleatorio Puro (sin sesgo, cada número tiene la misma probabilidad)")
        
        opcion_estrategia = input("Seleccione una estrategia (a/b/c/d/e): ").strip().lower()
        
        if opcion_estrategia == "a":
            strategy = "modelo"
        elif opcion_estrategia == "b":
            strategy = "calientes"
        elif opcion_estrategia == "c":
            strategy = "frios"
        elif opcion_estrategia == "d":
            strategy = "balanceado"
        elif opcion_estrategia == "e":
            strategy = "aleatorio"
        else:
            print("❌ Error: Opción de estrategia inválida.")
            continue
        break
    
    return dia_a_predecir, strategy

def main():
    """
    Función principal que orquesta el proceso de carga de datos, entrenamiento y predicción.
    """
    print("\n📅 Quiniela Predictor Mejorado 📅")
    print("="*60)
    
    try:
        predictor = QuinielaPredictor() # Instanciar la clase del predictor
        
        # Cargar datos históricos
        print("\n📂 Cargando datos históricos desde 'historicoquiniela.xlsx'...")
        df_raw = predictor.load_data("historicoquiniela.xlsx")
        
        if df_raw.empty:
            raise ValueError("No se encontraron datos válidos en 'historicoquiniela.xlsx'. Asegúrese de que el archivo existe y tiene el formato correcto (8 columnas, números 0-36).")
        
        print(f"\n📆 Último día con datos: {predictor.ultimo_dia.capitalize()}")
        siguiente_dia = predictor.get_next_day()
        
        # Obtener el día y la estrategia de predicción del usuario
        dia_a_predecir, strategy = get_user_choice(siguiente_dia, predictor)
        
        # Procesamiento de datos
        print("\n🔄 Transformando datos a formato largo...")
        df_long = predictor.transform_to_long_format(df_raw)
        print(f"\n📊 Total de registros procesados: {len(df_long)}")
        
        # Calcular estadísticas de números (calientes, fríos, frecuentes)
        print("\n🔧 Calculando estadísticas de números (calientes, fríos, frecuentes)...")
        predictor.calculate_number_stats(df_long)
        
        # Crear características para el modelo
        print("\n🔧 Creando características para el modelo...")
        df_processed = predictor.create_features(df_long)
        
        # Entrenamiento del modelo
        print("\n🚀 Entrenando modelo predictivo (si hay suficientes datos)...")
        predictor.train_model(df_processed)
        
        # Generar predicciones
        print(f"\n🎯 Generando predicciones para {dia_a_predecir.capitalize()} usando la estrategia '{strategy}'...")
        resultados = predictor.predict_day(dia_a_predecir, strategy)
        
        # Mostrar resultados
        print("\n📋 Resultados de la Predicción:")
        print(resultados.to_string(index=False))
        
        # Guardar resultados en un archivo Excel
        archivo_salida = f"prediccion_{dia_a_predecir}_{strategy}.xlsx"
        resultados.to_excel(archivo_salida, index=False)
        print(f"\n💾 Resultados guardados en '{archivo_salida}'")
        print("\nRecuerda: La quiniela es un juego de azar. ¡Mucha suerte!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n❌ Error: El archivo 'historicoquiniela.xlsx' no se encontró. Asegúrese de que el archivo está en el mismo directorio que el script.")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Ha ocurrido un error inesperado: {str(e)}")
        print("="*60)

if __name__ == "__main__":
    main()