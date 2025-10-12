import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import warnings
from collections import Counter
import joblib
import os
from datetime import datetime

# Nuevas importaciones para la Red Neuronal
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from sklearn.preprocessing import LabelEncoder

# Configuración global
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

class QuinielaPredictor:
    def __init__(self, model_dir="saved_models"):
        self.model = None
        self.encoder = None
        self.num_classes = 0
        self.model_trained = False
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.columnas_dias = []
        self.ultimo_dia = None
        self.test_size = 0.2
        self.random_state = 42
        self.all_possible_numbers = list(range(0, 37))
        self.hot_numbers = []
        self.cold_numbers = []
        self.most_frequent_overall = []
        self.probabilidades = {}
        self.X_test = None
        self.y_test = None
        self.y_pred_test = None
        self.metrics_results = {}
        
        # Nuevos atributos para persistencia
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "quiniela_model.h5")
        self.encoder_path = os.path.join(model_dir, "encoder.pkl")
        self.metadata_path = os.path.join(model_dir, "training_metadata.pkl")
        
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self):
        """Guarda el modelo y el encoder entrenados"""
        if self.model_trained and self.model is not None and not isinstance(self.model, str):
            try:
                # Guardar modelo de Keras
                self.model.save(self.model_path)
                
                # Guardar encoder y metadata
                joblib.dump({
                    'encoder': self.encoder,
                    'num_classes': self.num_classes,
                    'hot_numbers': self.hot_numbers,
                    'cold_numbers': self.cold_numbers,
                    'most_frequent_overall': self.most_frequent_overall,
                    'last_training': datetime.now(),
                    'metrics': self.metrics_results,
                    'columnas_dias': self.columnas_dias,
                    'ultimo_dia': self.ultimo_dia
                }, self.encoder_path)
                
                print(f"💾 Modelo guardado en {self.model_path}")
                return True
            except Exception as e:
                print(f"⚠️ Error guardando modelo: {str(e)}")
                return False
        return False

    def load_model(self):
        """Carga un modelo previamente entrenado"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                # Cargar modelo de Keras
                self.model = load_model(self.model_path)
                
                # Cargar encoder y metadata
                metadata = joblib.load(self.encoder_path)
                self.encoder = metadata['encoder']
                self.num_classes = metadata['num_classes']
                self.hot_numbers = metadata.get('hot_numbers', [])
                self.cold_numbers = metadata.get('cold_numbers', [])
                self.most_frequent_overall = metadata.get('most_frequent_overall', [])
                self.columnas_dias = metadata.get('columnas_dias', [])
                self.ultimo_dia = metadata.get('ultimo_dia', None)
                self.model_trained = True
                
                last_training = metadata.get('last_training', 'Desconocido')
                if isinstance(last_training, datetime):
                    last_training = last_training.strftime("%Y-%m-%d %H:%M")
                
                print(f"🔍 Modelo cargado (entrenado: {last_training})")
                return True
        except Exception as e:
            print(f"⚠️ Error cargando modelo: {str(e)}")
        
        return False

    def load_data(self, filepath):
        """Carga datos con manejo robusto de encabezados"""
        try:
            df = pd.read_excel(filepath, header=None)
            
            if df.shape[1] < 8:
                raise ValueError("El archivo debe tener al menos 8 columnas (hora + 7 días)")
            
            df.columns = ['hora'] + [f'dia_{i}' for i in range(1, df.shape[1])]
            self.columnas_dias = df.columns[1:]
            
            for i in reversed(range(len(self.columnas_dias))):
                if not df.iloc[:, i+1].isnull().all():
                    self.ultimo_dia = self.dias_semana[i % 7]
                    break
            
            if self.ultimo_dia is None:
                raise ValueError("No se encontraron datos válidos en ninguna columna")
            
            for col in self.columnas_dias:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: x if pd.isna(x) or (0 <= x <= 36) else np.nan)
            
            df = df.dropna(subset=self.columnas_dias, how='all')
            df = df.dropna(subset=['hora'])
            
            print(f"DEBUG: Datos cargados - {len(df)} filas válidas")
            return df
            
        except Exception as e:
            print(f"ERROR: No se pudo cargar el archivo: {str(e)}")
            return pd.DataFrame()

    def get_next_day(self):
        """Calcula el próximo día cíclicamente"""
        if self.ultimo_dia is None:
            raise ValueError("Primero cargue los datos")
            
        idx = self.dias_semana.index(self.ultimo_dia)
        return self.dias_semana[(idx + 1) % 7]

    def calculate_number_stats(self, df):
        """Calcula estadísticas de números"""
        if df.empty:
            print("No hay datos para calcular estadísticas")
            return
            
        all_nums = []
        for col in self.columnas_dias:
            all_nums.extend(df[col].dropna().astype(int).tolist())
        
        if not all_nums:
            print("No se encontraron números válidos")
            return
            
        counts = Counter(all_nums)
        self.most_frequent_overall = [num for num, _ in counts.most_common(8)]
        
        recent_nums = []
        for _, row in df.tail(100).iterrows():
            for col in self.columnas_dias:
                if pd.notna(row[col]):
                    recent_nums.append(int(row[col]))
        
        recent_counts = Counter(recent_nums)
        self.hot_numbers = [num for num, _ in recent_counts.most_common(5)]
        self.cold_numbers = list(set(self.all_possible_numbers) - set(recent_nums))
        
        print(f"\nEstadísticas calculadas:")
        print(f"Frecuentes: {self.most_frequent_overall}")
        print(f"Calientes: {self.hot_numbers}")
        print(f"Fríos: {self.cold_numbers[:10]}... (total: {len(self.cold_numbers)})")

    def evaluate_model(self):
        """Evalúa el modelo entrenado"""
        if not self.model_trained or self.y_test is None or self.y_pred_test is None:
            print("No se puede evaluar el modelo - no hay datos de prueba")
            self.metrics_results = {"Mensaje": "Modelo no evaluado"}
            return
            
        try:
            accuracy = accuracy_score(self.y_test, self.y_pred_test)
            mae = mean_absolute_error(self.y_test, self.y_pred_test)
            
            self.metrics_results = {
                "Exactitud": f"{accuracy:.2%}",
                "Error Absoluto Medio": f"{mae:.2f}",
                "Muestra de Prueba": len(self.y_test)
            }
            
            print("\n📊 Métricas del modelo:")
            for k, v in self.metrics_results.items():
                print(f"- {k}: {v}")
                
        except Exception as e:
            print(f"Error evaluando modelo: {str(e)}")
            self.metrics_results = {"Error": str(e)}

    def train_probabilistic_model(self, df_long):
        """Entrena un modelo basado en frecuencias"""
        self.model = "probabilistico"
        self.model_trained = True
        
        self.probabilidades = df_long.groupby(['hora', 'dia_num'])['numero'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()
        
        print("Modelo probabilístico entrenado (basado en frecuencias)")
        self.metrics_results = {"Modelo": "Probabilístico (frecuencias)"}
        return self.model

    def train_model(self, df, force_retrain=False):
        """Entrena el modelo o carga uno existente"""
        
        # Intentar cargar modelo existente si no forzamos reentrenamiento
        if not force_retrain and self.load_model():
            print("✅ Modelo pre-entrenado cargado exitosamente")
            return self.model
        
        print("🔄 Entrenando nuevo modelo...")
        
        records = []
        for _, row in df.iterrows():
            for i, col in enumerate(self.columnas_dias):
                num = row[col]
                if pd.notna(num):
                    dia_num = i % 7
                    records.append({
                        'hora': int(row['hora']),
                        'numero': int(num),
                        'dia_num': dia_num,
                        'es_finde': int(dia_num >= 5)
                    })
        
        if not records:
            print("No hay datos para entrenar")
            return self.train_probabilistic_model(pd.DataFrame())
            
        df_long = pd.DataFrame(records)
        df_long['hora_dia'] = df_long['hora'] * (df_long['dia_num'] + 1)
        
        if len(df_long) < 50:
            print("Pocos datos. Usando modelo probabilístico")
            return self.train_probabilistic_model(df_long)
            
        try:
            X = df_long[['hora', 'dia_num', 'hora_dia', 'es_finde']]
            y = df_long['numero']
            
            # Preparación de datos para la Red Neuronal
            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)
            self.num_classes = len(self.encoder.classes_)

            X_train, self.X_test, y_train_cat, self.y_test_cat = train_test_split(
                X, y_categorical, test_size=self.test_size, random_state=self.random_state
            )
            
            self.y_test = self.encoder.inverse_transform(np.argmax(self.y_test_cat, axis=1))

            # Callbacks para mejorar el entrenamiento
            checkpoint = ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            )
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )

            # Definición y Entrenamiento de la Red Neuronal
            input_shape = X_train.shape[1]
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(self.num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print("\n⚙️ Entrenando Red Neuronal...")
            history = model.fit(
                X_train, y_train_cat,
                epochs=100,  # Más épocas con early stopping
                batch_size=32,
                validation_split=0.1,
                callbacks=[checkpoint, early_stop],
                verbose=0
            )
            
            # Cargar el mejor modelo guardado durante el entrenamiento
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print("✅ Mejor modelo cargado desde checkpoint")
            else:
                self.model = model
                print("✅ Modelo final entrenado")
                
            self.model_trained = True
            
            # Evaluar y guardar
            y_pred_proba = self.model.predict(self.X_test, verbose=0)
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)
            self.y_pred_test = self.encoder.inverse_transform(y_pred_encoded)
            
            self.evaluate_model()
            self.save_model()  # Guardar modelo final
            
            print(f"✅ Entrenamiento completado - Épocas efectivas: {len(history.history['loss'])}")
            return self.model
            
        except Exception as e:
            print(f"Error entrenando modelo: {str(e)}")
            return self.train_probabilistic_model(df_long)

    def predict_day(self, dia_semana, strategy="modelo"):
        """Genera predicciones para un día específico"""
        try:
            dia_num = self.dias_semana.index(dia_semana.lower())
        except ValueError:
            raise ValueError(f"Día debe ser uno de: {', '.join(self.dias_semana)}")
            
        horas = [10, 11, 12, 13, 14, 15]
        predicciones = []
        
        for hora in horas:
            num = None
            if strategy == "modelo" and self.model_trained and not isinstance(self.model, str):
                try:
                    X_new = pd.DataFrame([{
                        'hora': hora,
                        'dia_num': dia_num,
                        'hora_dia': hora * (dia_num + 1),
                        'es_finde': int(dia_num >= 5)
                    }])
                    
                    proba = self.model.predict(X_new, verbose=0)[0]
                    nums = self.encoder.classes_
                    num = np.random.choice(nums, p=proba)
                except Exception as e:
                    print(f"DEBUG: Fallo en predicción NN: {e}")
                    num = np.random.choice(self.all_possible_numbers)
                    
            elif strategy == "calientes":
                num = np.random.choice(self.hot_numbers if self.hot_numbers else self.all_possible_numbers)
                
            elif strategy == "frios":
                num = np.random.choice(self.cold_numbers if self.cold_numbers else self.all_possible_numbers)
                
            elif strategy == "balanceado":
                combined = list(set(self.hot_numbers + self.cold_numbers + self.most_frequent_overall))
                num = np.random.choice(combined if combined else self.all_possible_numbers)
                
            else: # Aleatorio
                num = np.random.choice(self.all_possible_numbers)
            
            predicciones.append(f"{num:02d}")
            
        if len(set(predicciones)) < len(horas)/2:
            unique = list(set(predicciones))
            extra = np.random.choice(
                [n for n in self.all_possible_numbers if f"{n:02d}" not in unique],
                size=len(horas)-len(unique),
                replace=False
            )
            predicciones = unique + [f"{n:02d}" for n in extra]
            np.random.shuffle(predicciones)
            
        return pd.DataFrame({
            'Hora': [f"{h}:00" for h in horas],
            'Predicción': predicciones,
            'Día': dia_semana.capitalize(),
            'Estrategia': strategy
        })

    def generate_mixed_numbers(self, dia_semana, total_numbers=12):
        """Genera 12 números mezclando las 5 estrategias disponibles"""
        try:
            dia_num = self.dias_semana.index(dia_semana.lower())
        except ValueError:
            raise ValueError(f"Día debe ser uno de: {', '.join(self.dias_semana)}")
        
        print(f"\n🎲 Generando {total_numbers} números mezclando todas las estrategias...")
        
        # Definir cuántos números generar por cada estrategia
        numbers_per_strategy = total_numbers // 5
        extra_numbers = total_numbers % 5
        
        mixed_numbers = []
        
        # Estrategias disponibles
        estrategias = ['modelo', 'calientes', 'frios', 'balanceado', 'aleatorio']
        
        for i, estrategia in enumerate(estrategias):
            # Para la última estrategia, agregar los números extra
            count = numbers_per_strategy + (1 if i == len(estrategias) - 1 else 0) if extra_numbers > 0 else numbers_per_strategy
            
            if i == 0:  # Para modelo, usar horas específicas
                horas = list(range(10, 10 + count))
            else:
                horas = [10] * count  # Hora fija para otras estrategias
            
            for hora in horas:
                num = None
                if estrategia == "modelo" and self.model_trained and not isinstance(self.model, str):
                    try:
                        X_new = pd.DataFrame([{
                            'hora': hora,
                            'dia_num': dia_num,
                            'hora_dia': hora * (dia_num + 1),
                            'es_finde': int(dia_num >= 5)
                        }])
                        
                        proba = self.model.predict(X_new, verbose=0)[0]
                        nums = self.encoder.classes_
                        num = np.random.choice(nums, p=proba)
                    except Exception as e:
                        num = np.random.choice(self.all_possible_numbers)
                        
                elif estrategia == "calientes":
                    num = np.random.choice(self.hot_numbers if self.hot_numbers else self.all_possible_numbers)
                    
                elif estrategia == "frios":
                    num = np.random.choice(self.cold_numbers if self.cold_numbers else self.all_possible_numbers)
                    
                elif estrategia == "balanceado":
                    combined = list(set(self.hot_numbers + self.cold_numbers + self.most_frequent_overall))
                    num = np.random.choice(combined if combined else self.all_possible_numbers)
                    
                else: # Aleatorio
                    num = np.random.choice(self.all_possible_numbers)
                
                mixed_numbers.append({
                    'Número': f"{num:02d}",
                    'Estrategia': estrategia,
                    'Hora': f"{hora}:00" if estrategia == "modelo" else "Variable"
                })
        
        # Mezclar los números para que no estén agrupados por estrategia
        np.random.shuffle(mixed_numbers)
        
        # Asegurar que no haya duplicados
        unique_numbers = []
        seen_numbers = set()
        
        for num_data in mixed_numbers:
            if num_data['Número'] not in seen_numbers:
                unique_numbers.append(num_data)
                seen_numbers.add(num_data['Número'])
        
        # Si hay menos de 12 números únicos, completar con números aleatorios
        while len(unique_numbers) < total_numbers:
            new_num = f"{np.random.choice(self.all_possible_numbers):02d}"
            if new_num not in seen_numbers:
                unique_numbers.append({
                    'Número': new_num,
                    'Estrategia': 'complementario',
                    'Hora': 'Variable'
                })
                seen_numbers.add(new_num)
        
        # Crear DataFrame con los resultados
        df_resultados = pd.DataFrame(unique_numbers[:total_numbers])
        
        # Mostrar resumen por estrategia
        print(f"\n📊 Resumen por estrategia:")
        summary = df_resultados['Estrategia'].value_counts()
        for estrategia, count in summary.items():
            print(f"  - {estrategia.capitalize()}: {count} números")
        
        return df_resultados

def main():
    print("\n🔮 Quiniela Predictor - Con Modelo Persistente 🔮")
    print("="*50)
    
    predictor = QuinielaPredictor()
    
    try:
        print("\n📂 Cargando datos...")
        df = predictor.load_data("historicoquiniela.xlsx")
        
        if df.empty:
            raise ValueError("No se pudieron cargar datos válidos")
            
        print(f"\n📆 Último día con datos: {predictor.ultimo_dia.capitalize()}")
        siguiente_dia = predictor.get_next_day()
        
        print("\nOpciones de día:")
        print(f"1. Predecir {siguiente_dia.capitalize()} (siguiente día)")
        print("2. Elegir otro día")
        opcion = input("Seleccione (1/2): ").strip()
        
        if opcion == "1":
            dia = siguiente_dia
        else:
            while True:
                dia = input(f"Ingrese día ({', '.join(predictor.dias_semana)}): ").lower()
                if dia in predictor.dias_semana:
                    break
                print("Día inválido. Intente nuevamente.")
        
        # Preguntar si forzar reentrenamiento
        print("\n🔄 Opciones de entrenamiento:")
        print("1. Usar modelo existente (recomendado)")
        print("2. Forzar reentrenamiento completo")
        entrenamiento_opcion = input("Seleccione (1/2): ").strip()
        force_retrain = (entrenamiento_opcion == "2")
        
        print("\nEstrategias disponibles:")
        print("a. Modelo predictivo")
        print("b. Números calientes")
        print("c. Números fríos")
        print("d. Balanceado")
        print("e. Aleatorio")
        print("f. MEZCLA (12 números combinando todas las estrategias)")
        estrategia_opcion = input("Seleccione estrategia (a-f): ").lower().strip()
        
        estrategias = {
            'a': 'modelo',
            'b': 'calientes',
            'c': 'frios',
            'd': 'balanceado',
            'e': 'aleatorio',
            'f': 'mezcla'
        }
        estrategia = estrategias.get(estrategia_opcion, 'aleatorio')
        
        print("\n🔧 Calculando estadísticas...")
        predictor.calculate_number_stats(df)
        
        print("\n🤖 Configurando modelo...")
        predictor.train_model(df, force_retrain=force_retrain)
        
        if estrategia == 'mezcla':
            print(f"\n🎯 Generando 12 números mezclados para {dia.capitalize()}...")
            resultados = predictor.generate_mixed_numbers(dia, 12)
            
            print("\n📋 Resultados (12 números mezclados):")
            print(resultados.to_string(index=False))
            
            try:
                archivo = f"prediccion_mezcla_{dia}.xlsx"
                resultados.to_excel(archivo, index=False)
                print(f"\n💾 Resultados guardados en '{archivo}'")
            except Exception as e:
                print(f"\n⚠️ No se pudo guardar el archivo: {str(e)}")
        else:
            print(f"\n🎯 Generando predicciones para {dia.capitalize()}...")
            resultados = predictor.predict_day(dia, estrategia)
            
            print("\n📋 Resultados:")
            print(resultados.to_string(index=False))
            
            try:
                archivo = f"prediccion_{dia}_{estrategia}.xlsx"
                resultados.to_excel(archivo, index=False)
                print(f"\n💾 Resultados guardados en '{archivo}'")
            except Exception as e:
                print(f"\n⚠️ No se pudo guardar el archivo: {str(e)}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n¡Buena suerte!")
    print("="*50)

if __name__ == "__main__":
    main()