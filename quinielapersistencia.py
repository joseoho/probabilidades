import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
from collections import Counter, defaultdict
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # pyright: ignore[reportMissingModuleSource]

try:
    # Nuevas importaciones para la Red Neuronal
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
    from tensorflow.keras.utils import to_categorical # type: ignore
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    from tensorflow.keras.regularizers import l2 # type: ignore


except ImportError:
    print("⚠️ TensorFlow no está disponible. Algunas funciones no funcionarán.")

warnings.filterwarnings('ignore')

class PredictorAvanzado:
    def __init__(self, model_dir="saved_models_avanzados"):
        self.model = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.num_classes = 0
        self.model_trained = False
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.columnas_dias = []
        self.ultimo_dia = None
        self.all_possible_numbers = list(range(0, 37))
        
        # Atributos para predicciones avanzadas
        self.ensemble_models = {}
        self.series_temporales = {}
        self.clusters_numeros = {}
        self.feature_columns_fit = []
        self.historial_predicciones = []
        
        # Configuración de paths
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self, filepath):
        """Carga datos con manejo robusto de encabezados"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
            
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, header=None, engine='openpyxl')
            elif filepath.endswith('.xls'):
                df = pd.read_excel(filepath, header=None)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath, header=None)
            else:
                raise ValueError("Formato de archivo no soportado")
            
            if df.shape[1] < 8:
                raise ValueError("El archivo debe tener al menos 8 columnas (hora + 7 días)")
            
            # Asignar nombres de columnas
            df.columns = ['hora'] + [f'dia_{i}' for i in range(1, df.shape[1])]
            self.columnas_dias = df.columns[1:]
            
            # Procesar hora
            df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
            df = df.dropna(subset=['hora'])
            df['hora'] = df['hora'].astype(int)
            
            # Validar y limpiar datos numéricos
            for col in self.columnas_dias:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: x if pd.isna(x) or (0 <= x <= 36) else np.nan)
            
            # Determinar último día con datos
            self.ultimo_dia = None
            for i in reversed(range(len(self.columnas_dias))):
                if not df.iloc[:, i+1].isnull().all():
                    self.ultimo_dia = self.dias_semana[i % 7]
                    break
            
            if self.ultimo_dia is None:
                raise ValueError("No se encontraron datos válidos en ninguna columna")
            
            # Limpieza final
            df = df.dropna(subset=self.columnas_dias, how='all')
            df = df.reset_index(drop=True)
            
            print(f"✅ Datos cargados: {len(df)} filas, {len(self.columnas_dias)} días")
            print(f"📅 Rango de horas: {df['hora'].min()}-{df['hora'].max()}")
            print(f"📊 Último día con datos: {self.ultimo_dia}")
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR cargando datos: {str(e)}")
            return pd.DataFrame()

    def get_next_day(self):
        """Calcula el próximo día cíclicamente"""
        if self.ultimo_dia is None:
            raise ValueError("Primero cargue los datos")
            
        idx = self.dias_semana.index(self.ultimo_dia)
        return self.dias_semana[(idx + 1) % 7]

    def crear_caracteristicas_avanzadas(self, df_long):
        """Crea características avanzadas para mejorar las predicciones"""
        
        print("🔄 Creando características avanzadas...")
        
        # Características básicas mejoradas
        df_long['hora_dia'] = df_long['hora'] * (df_long['dia_num'] + 1)
        df_long['es_finde'] = df_long['dia_num'].apply(lambda x: 1 if x >= 5 else 0)
        df_long['hora_tardia'] = df_long['hora'].apply(lambda x: 1 if x >= 14 else 0)
        
        # Características cíclicas para hora y día
        df_long['hora_sin'] = np.sin(2 * np.pi * df_long['hora'] / 24)
        df_long['hora_cos'] = np.cos(2 * np.pi * df_long['hora'] / 24)
        df_long['dia_sin'] = np.sin(2 * np.pi * df_long['dia_num'] / 7)
        df_long['dia_cos'] = np.cos(2 * np.pi * df_long['dia_num'] / 7)
        
        # Frecuencias y estadísticas
        self._agregar_estadisticas_frecuencia(df_long)
        
        # Características de tendencia (simplificadas para evitar problemas)
        self._agregar_tendencias_simplificadas(df_long)
        
        # Características de clustering
        self._agregar_clusters(df_long)
        
        # Definir las columnas de características que se usarán consistentemente
        self.feature_columns_fit = [
            'hora', 'dia_num', 'hora_dia', 'es_finde', 'hora_tardia',
            'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos',
            'freq_global', 'freq_hora', 'freq_dia', 'cluster'
        ]
        
        print(f"✅ Características creadas. {len(self.feature_columns_fit)} features disponibles")
        return df_long

    def _agregar_estadisticas_frecuencia(self, df_long):
        """Agrega estadísticas de frecuencia avanzadas"""
        
        # Frecuencia global por número
        freq_global = df_long['numero'].value_counts(normalize=True).to_dict()
        df_long['freq_global'] = df_long['numero'].map(freq_global).fillna(0)
        
        # Frecuencia por hora
        for hora in range(10, 16):
            mask = df_long['hora'] == hora
            if mask.any():
                freq_hora = df_long[mask]['numero'].value_counts(normalize=True).to_dict()
                df_long.loc[mask, 'freq_hora'] = df_long.loc[mask, 'numero'].map(freq_hora).fillna(0)
        
        # Frecuencia por día
        for dia in range(7):
            mask = df_long['dia_num'] == dia
            if mask.any():
                freq_dia = df_long[mask]['numero'].value_counts(normalize=True).to_dict()
                df_long.loc[mask, 'freq_dia'] = df_long.loc[mask, 'numero'].map(freq_dia).fillna(0)

        # Rellenar valores NaN
        df_long['freq_global'] = df_long['freq_global'].fillna(0)
        df_long['freq_hora'] = df_long['freq_hora'].fillna(0)
        df_long['freq_dia'] = df_long['freq_dia'].fillna(0)

    def _agregar_tendencias_simplificadas(self, df_long):
        """Calcula tendencias temporales simplificadas"""
        
        # Ordenar para análisis temporal
        df_long = df_long.sort_values(['dia_num', 'hora']).reset_index(drop=True)
        
        # Característica simple: conteo de apariciones recientes
        df_long['conteo_reciente'] = df_long.groupby('numero').cumcount() + 1
        
        # Cluster básico basado en frecuencia
        freq_cluster = df_long.groupby('numero')['numero'].count()
        df_long['freq_cluster'] = df_long['numero'].map(freq_cluster).fillna(0)
        
        # Normalizar el cluster
        max_freq = df_long['freq_cluster'].max()
        if max_freq > 0:
            df_long['freq_cluster'] = df_long['freq_cluster'] / max_freq

    def _agregar_clusters(self, df_long):
        """Agrupa números por comportamiento similar usando clustering"""
        
        try:
            # Características para clustering
            features_cluster = df_long.groupby('numero').agg({
                'hora': ['mean', 'std'],
                'dia_num': ['mean', 'std'],
                'freq_global': 'mean'
            }).fillna(0)
            
            features_cluster.columns = ['_'.join(col).strip() for col in features_cluster.columns]
            
            # Aplicar KMeans solo si hay suficientes números
            if len(features_cluster) >= 3:
                n_clusters = min(5, len(features_cluster))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features_cluster)
                
                # Mapear clusters a números
                self.clusters_numeros = dict(zip(features_cluster.index, clusters))
                df_long['cluster'] = df_long['numero'].map(self.clusters_numeros).fillna(-1)
            else:
                df_long['cluster'] = 0
                
            print("✅ Clustering completado")
                
        except Exception as e:
            print(f"⚠️ Error en clustering: {e}")
            df_long['cluster'] = 0

    def entrenar_modelo_ensemble(self, X, y):
        """Entrena múltiples modelos para ensemble learning"""
        
        print("🔄 Entrenando modelos de ensemble...")
        
        # Asegurar que solo usamos las columnas definidas
        X = X[self.feature_columns_fit]
        
        # Codificar target
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        self.num_classes = len(self.encoder.classes_)
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Modelo 1: Red Neuronal Avanzada
        try:
            nn_model = self._crear_modelo_red_neuronal(X_scaled.shape[1])
            y_categorical = to_categorical(y_encoded)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            history = nn_model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=0
            )
            
            self.ensemble_models['red_neuronal'] = nn_model
            print("✅ Red neuronal entrenada")
            
        except Exception as e:
            print(f"⚠️ Error en red neuronal: {e}")

        # Modelo 2: Random Forest
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            rf_model.fit(X_scaled, y_encoded)
            self.ensemble_models['random_forest'] = rf_model
            print("✅ Random Forest entrenado")
        except Exception as e:
            print(f"⚠️ Error en Random Forest: {e}")

        # Modelo 3: Gradient Boosting
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_scaled, y_encoded)
            self.ensemble_models['gradient_boosting'] = gb_model
            print("✅ Gradient Boosting entrenado")
        except Exception as e:
            print(f"⚠️ Error en Gradient Boosting: {e}")

        self.model_trained = True
        print(f"🎯 Ensemble entrenado con {len(self.ensemble_models)} modelos")
        return self.ensemble_models

    def _crear_modelo_red_neuronal(self, input_shape):
        """Crea una red neuronal avanzada para predicciones"""
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def predecir_ensemble(self, features, estrategia='ponderada'):
        """Realiza predicciones usando ensemble learning"""
        
        if not self.ensemble_models:
            raise ValueError("No hay modelos entrenados para ensemble")
        
        # Asegurar que las características tengan las mismas columnas que durante el entrenamiento
        features = features[self.feature_columns_fit]
        
        # Escalar características
        features_scaled = self.scaler.transform(features)
        
        predicciones = {}
        confianzas = {}
        
        # Predicciones de cada modelo
        for nombre, modelo in self.ensemble_models.items():
            try:
                if nombre == 'red_neuronal':
                    proba = modelo.predict(features_scaled, verbose=0)[0]
                    pred_idx = np.argmax(proba)
                    predicciones[nombre] = self.encoder.inverse_transform([pred_idx])[0]
                    confianzas[nombre] = np.max(proba)
                else:
                    proba = modelo.predict_proba(features_scaled)[0]
                    pred_idx = np.argmax(proba)
                    predicciones[nombre] = self.encoder.inverse_transform([pred_idx])[0]
                    confianzas[nombre] = np.max(proba)
            except Exception as e:
                print(f"⚠️ Error en modelo {nombre}: {e}")
                continue
        
        if not predicciones:
            raise ValueError("Ningún modelo pudo hacer predicciones")
        
        # Combinar predicciones según estrategia
        if estrategia == 'votacion':
            # Votación por mayoría
            counts = Counter(predicciones.values())
            prediccion_final = counts.most_common(1)[0][0]
            confianza_final = counts.most_common(1)[0][1] / len(predicciones)
            
        elif estrategia == 'ponderada':
            # Promedio ponderado por confianza
            votos_ponderados = defaultdict(float)
            for modelo, pred in predicciones.items():
                votos_ponderados[pred] += confianzas[modelo]
            
            prediccion_final = max(votos_ponderados.items(), key=lambda x: x[1])[0]
            confianza_final = votos_ponderados[prediccion_final] / sum(confianzas.values())
            
        else:  # 'max_confianza'
            # Modelo con mayor confianza
            mejor_modelo = max(confianzas.items(), key=lambda x: x[1])[0]
            prediccion_final = predicciones[mejor_modelo]
            confianza_final = confianzas[mejor_modelo]
        
        return {
            'prediccion': prediccion_final,
            'confianza': min(confianza_final, 1.0),  # 🔥 LIMITAR A 100%
            'detalle_modelos': predicciones,
            'confianzas_individuales': confianzas
        }

    def analizar_serie_temporal(self, df_long):
        """Analiza patrones de series temporales avanzados"""
        
        print("📈 Analizando series temporales...")
        
        # Preparar datos para análisis temporal
        df_temp = df_long.copy()
        
        # Crear timestamp simulado para análisis
        base_date = datetime(2024, 1, 1)
        df_temp['dias_desde_base'] = df_temp['dia_num'] + (df_temp['hora'] / 24.0)
        df_temp = df_temp.sort_values('dias_desde_base')
        
        # Análisis por número
        self.series_temporales = {}
        numeros_analizados = min(20, len(df_temp['numero'].unique()))
        
        for numero in list(df_temp['numero'].unique())[:numeros_analizados]:
            mask = df_temp['numero'] == numero
            serie_numeros = df_temp[mask]
            
            if len(serie_numeros) > 3:
                # Calcular estadísticas temporales básicas
                dias_series = serie_numeros['dias_desde_base']
                if len(dias_series) > 1:
                    intervalos = dias_series.diff().dropna()
                    intervalo_promedio = intervalos.mean() if not intervalos.empty else 0
                    
                    self.series_temporales[numero] = {
                        'frecuencia': len(serie_numeros),
                        'intervalo_promedio': intervalo_promedio,
                        'ultima_aparicion': dias_series.max(),
                        'tendencia': 'creciente' if len(serie_numeros) > 10 else 'estable'
                    }
        
        print(f"✅ Análisis temporal completado para {len(self.series_temporales)} números")

    def predecir_con_serie_temporal(self, dia_semana, hora):
        """Predicciones basadas en análisis de series temporales"""
        
        dia_num = self.dias_semana.index(dia_semana.lower())
        current_time_fake = dia_num + (hora / 24.0)  # Tiempo simulado
        
        # Filtrar números con patrones temporales relevantes
        numeros_candidatos = []
        
        for numero, info in self.series_temporales.items():
            # Calcular probabilidad basada en patrón temporal
            if info['intervalo_promedio'] > 0:
                tiempo_desde_ultima = current_time_fake - info['ultima_aparicion']
                factor_tiempo = max(0, 1 - abs(tiempo_desde_ultima - info['intervalo_promedio']) / info['intervalo_promedio'])
                probabilidad = min((info['frecuencia'] / 100) * factor_tiempo, 0.5)  # 🔥 LIMITAR A 50%
                
                if probabilidad > 0.01:
                    numeros_candidatos.append({
                        'numero': numero,
                        'probabilidad': probabilidad,
                        'razon': f"Patrón temporal (frecuencia: {info['frecuencia']})"
                    })
        
        return sorted(numeros_candidatos, key=lambda x: x['probabilidad'], reverse=True)[:5]

    def predecir_con_monte_carlo(self, dia_semana, hora, n_simulaciones=200):
        """Predicciones usando simulación Monte Carlo"""
        
        dia_num = self.dias_semana.index(dia_semana.lower())
        
        # Características base para la simulación
        features_base = self._preparar_features_base(dia_semana, hora)
        
        # Simular variaciones
        resultados_simulacion = []
        
        for _ in range(n_simulaciones):
            # Agregar ruido aleatorio a las características
            features_simuladas = features_base.copy()
            features_simuladas['hora'] = max(10, min(15, features_simuladas['hora'] + np.random.normal(0, 0.3)))
            features_simuladas['freq_global'] = np.random.uniform(0.01, 0.05)
            features_simuladas['freq_hora'] = np.random.uniform(0.01, 0.05)
            features_simuladas['freq_dia'] = np.random.uniform(0.01, 0.05)
            
            # Usar ensemble para predicción si está disponible
            try:
                df_features = pd.DataFrame([features_simuladas])
                prediccion = self.predecir_ensemble(df_features)
                resultados_simulacion.append(prediccion['prediccion'])
            except:
                # Fallback a predicción aleatoria
                resultados_simulacion.append(np.random.choice(self.all_possible_numbers))
        
        # Analizar resultados de la simulación
        if resultados_simulacion:
            counts = Counter(resultados_simulacion)
            total = len(resultados_simulacion)
            
            predicciones_probables = []
            for numero, count in counts.most_common(8):
                probabilidad = min(count / total, 0.8)  # 🔥 LIMITAR A 80%
                predicciones_probables.append({
                    'numero': numero,
                    'probabilidad': probabilidad,
                    'frecuencia_simulacion': count,
                    'metodo': 'Monte Carlo'
                })
            
            return predicciones_probables
        
        return []

    def _preparar_features_base(self, dia_semana, hora):
        """Prepara características base para predicción"""
        
        dia_num = self.dias_semana.index(dia_semana.lower())
        
        features = {
            'hora': hora,
            'dia_num': dia_num,
            'hora_dia': hora * (dia_num + 1),
            'es_finde': int(dia_num >= 5),
            'hora_tardia': 1 if hora >= 14 else 0,
            'hora_sin': np.sin(2 * np.pi * hora / 24),
            'hora_cos': np.cos(2 * np.pi * hora / 24),
            'dia_sin': np.sin(2 * np.pi * dia_num / 7),
            'dia_cos': np.cos(2 * np.pi * dia_num / 7),
            'freq_global': 0.02,
            'freq_hora': 0.02,
            'freq_dia': 0.02,
            'cluster': 0  # Valor por defecto
        }
        
        return features

    def _preparar_features_prediccion(self, dia_semana, hora):
        """Prepara características para predicción (versión DataFrame)"""
        
        features_base = self._preparar_features_base(dia_semana, hora)
        return pd.DataFrame([features_base])[self.feature_columns_fit]

    def _obtener_numero_diverso(self, predicciones_anteriores):
        """Obtiene un número que no se haya repetido mucho"""
        
        if not predicciones_anteriores:
            return np.random.choice(self.all_possible_numbers)
        
        numeros_previos = [int(pred['Predicción']) for pred in predicciones_anteriores]
        conteo_previo = Counter(numeros_previos)
        
        # Priorizar números no usados o poco usados
        numeros_no_usados = [num for num in self.all_possible_numbers if conteo_previo.get(num, 0) == 0]
        
        if numeros_no_usados:
            return np.random.choice(numeros_no_usados)
        else:
            # Usar el menos repetido
            numeros_poco_usados = [num for num, count in conteo_previo.items() if count <= 1]
            if numeros_poco_usados:
                return np.random.choice(numeros_poco_usados)
            else:
                return min(conteo_previo.items(), key=lambda x: x[1])[0]

    def _seleccionar_mejor_numero(self, numeros_ordenados, predicciones_anteriores):
        """Selecciona el mejor número evitando repeticiones excesivas"""
        
        if not predicciones_anteriores:
            return numeros_ordenados[0][0]  # Primer elemento si no hay historia
        
        # Contar apariciones previas del número
        numeros_previos = [int(pred['Predicción']) for pred in predicciones_anteriores]
        conteo_previo = Counter(numeros_previos)
        
        # Buscar números que no se han repetido
        for numero, confianza in numeros_ordenados:
            if conteo_previo.get(numero, 0) == 0:
                return numero
        
        # Buscar números que se han repetido solo una vez
        for numero, confianza in numeros_ordenados:
            if conteo_previo.get(numero, 0) == 1:
                return numero
        
        # Si todos se han repetido 2+ veces, devolver el más confiable pero diferente
        numeros_no_usados_recientemente = [
            num for num, conf in numeros_ordenados 
            if num not in numeros_previos[-3:]  # No usado en las últimas 3 predicciones
        ]
        
        if numeros_no_usados_recientemente:
            return numeros_no_usados_recientemente[0]
        
        # Último recurso: devolver el más confiable
        return numeros_ordenados[0][0]

    def _combinar_predicciones(self, todas_predicciones, horas):
        """Combina predicciones de múltiples métodos - VERSIÓN MEJORADA"""
        
        predicciones_finales = []
        
        for hora in horas:
            # Filtrar predicciones para esta hora
            preds_hora = [p for p in todas_predicciones if p['hora'] == hora]
            
            if not preds_hora:
                # Fallback a predicción diversa
                numero_fallback = self._obtener_numero_diverso(predicciones_finales)
                predicciones_finales.append({
                    'Hora': f"{hora}:00",
                    'Predicción': f"{numero_fallback:02d}",
                    'Confianza': '15.00%',
                    'Método': 'Diversificación',
                    'Números_Alternativos': []  # 🔥 CORREGIDO: Sin 's' al final
                })
                continue
            
            # Agrupar por número y sumar confianzas (CON NORMALIZACIÓN)
            numeros_confianza = defaultdict(float)
            metodos_por_numero = defaultdict(list)
            
            for pred in preds_hora:
                numero = pred['numero']
                confianza = min(pred.get('confianza', 0.1), 1.0)  # 🔥 LIMITAR A 100%
                numeros_confianza[numero] += confianza
                metodos_por_numero[numero].append(pred.get('metodo', 'Desconocido'))
            
            # 🔥 NORMALIZAR LAS CONFIANZAS para que sumen 1
            total_confianza = sum(numeros_confianza.values())
            if total_confianza > 0:
                for num in numeros_confianza:
                    numeros_confianza[num] = numeros_confianza[num] / total_confianza
            
            # Seleccionar número con mayor confianza combinada (CON DIVERSIDAD)
            if numeros_confianza:
                # Ordenar por confianza
                numeros_ordenados = sorted(numeros_confianza.items(), key=lambda x: x[1], reverse=True)
                
                # 🔥 EVITAR REPETIR EL MISMO NÚMERO MUCHAS VECES
                mejor_numero = self._seleccionar_mejor_numero(numeros_ordenados, predicciones_finales)
                
                # Preparar alternativas (máximo 2)
                alternativas = []
                for num, conf in numeros_ordenados[:4]:  # Tomar hasta 4 mejores
                    if num != mejor_numero and len(alternativas) < 2:
                        num_metodos = metodos_por_numero[num]
                        alternativas.append({
                            'numero': f"{num:02d}",
                            'confianza': f"{conf:.2%}",
                            'metodos': num_metodos
                        })
                
                metodos_mejor = metodos_por_numero[mejor_numero]
                confianza_mejor = numeros_confianza[mejor_numero]
                
                # 🔥 AJUSTAR CONFIANZA PARA SER REALISTA
                confianza_ajustada = min(confianza_mejor * 0.8, 0.85)  # Máximo 85% de confianza
                
                predicciones_finales.append({
                    'Hora': f"{hora}:00",
                    'Predicción': f"{mejor_numero:02d}",
                    'Confianza': f"{confianza_ajustada:.2%}",
                    'Método': '+'.join(sorted(set(metodos_mejor))[:2]),
                    'Números_Alternativos': alternativas  # 🔥 CORREGIDO: Sin 's' al final
                })
            else:
                # Fallback diversificado
                numero_fallback = self._obtener_numero_diverso(predicciones_finales)
                predicciones_finales.append({
                    'Hora': f"{hora}:00",
                    'Predicción': f"{numero_fallback:02d}",
                    'Confianza': '15.00%',
                    'Método': 'Diversificación',
                    'Números_Alternativos': []  # 🔥 CORREGIDO: Sin 's' al final
                })
        
        return pd.DataFrame(predicciones_finales)

    def generar_predicciones_inteligentes(self, dia_semana):
        """Genera predicciones usando múltiples métodos inteligentes - VERSIÓN MEJORADA"""
        
        print(f"\n🎯 Generando predicciones DIVERSIFICADAS para {dia_semana.capitalize()}...")
        
        horas = list(range(10, 16))
        todas_predicciones = []
        
        for hora in horas:
            print(f"   ⏰ Procesando hora {hora}:00...")
            
            # Método 1: Ensemble Learning
            try:
                features = self._preparar_features_prediccion(dia_semana, hora)
                pred_ensemble = self.predecir_ensemble(features)
                todas_predicciones.append({
                    'hora': hora,
                    'numero': pred_ensemble['prediccion'],
                    'confianza': pred_ensemble['confianza'],
                    'metodo': 'Ensemble',
                    'detalle': pred_ensemble
                })
            except Exception as e:
                print(f"      ⚠️ Ensemble falló: {e}")
            
            # Método 2: Series Temporales
            try:
                pred_temporal = self.predecir_con_serie_temporal(dia_semana, hora)
                if pred_temporal:
                    for pred in pred_temporal[:2]:  # 🔥 Tomar solo 2 mejores
                        todas_predicciones.append({
                            'hora': hora,
                            'numero': pred['numero'],
                            'confianza': pred['probabilidad'],
                            'metodo': 'Serie Temporal',
                            'detalle': pred
                        })
            except Exception as e:
                print(f"      ⚠️ Serie temporal falló: {e}")
            
            # Método 3: Monte Carlo
            try:
                pred_monte_carlo = self.predecir_con_monte_carlo(dia_semana, hora, 150)
                if pred_monte_carlo:
                    for pred in pred_monte_carlo[:2]:  # 🔥 Tomar solo 2 mejores
                        todas_predicciones.append({
                            'hora': hora,
                            'numero': pred['numero'],
                            'confianza': pred['probabilidad'],
                            'metodo': 'Monte Carlo',
                            'detalle': pred
                        })
            except Exception as e:
                print(f"      ⚠️ Monte Carlo falló: {e}")
        
        # Combinar y seleccionar mejores predicciones
        predicciones_finales = self._combinar_predicciones(todas_predicciones, horas)
        
        return predicciones_finales

# Función principal mejorada
def main_avanzada():
    print("\n🔮 PREDICTOR AVANZADO - PREDICCIONES DIVERSIFICADAS 🔮")
    print("="*70)
    
    predictor = PredictorAvanzado()
    
    try:
        # Cargar y preparar datos
        df = predictor.load_data("historicoquiniela.xlsx")
        
        if df.empty:
            raise ValueError("No se pudieron cargar datos válidos")
        
        # Preparar datos para entrenamiento
        records = []
        for _, row in df.iterrows():
            for i, col in enumerate(predictor.columnas_dias):
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
            raise ValueError("No se encontraron registros válidos para entrenar")
        
        df_long = pd.DataFrame(records)
        df_long = predictor.crear_caracteristicas_avanzadas(df_long)
        
        # Entrenar modelos
        X = df_long[predictor.feature_columns_fit]
        y = df_long['numero']
        
        predictor.entrenar_modelo_ensemble(X, y)
        predictor.analizar_serie_temporal(df_long)
        
        # Generar predicciones inteligentes
        dia_prediccion = input("\n📅 Ingrese el día a predecir: ").strip().lower()
        
        if dia_prediccion not in predictor.dias_semana:
            print("Día inválido. Usando siguiente día disponible.")
            dia_prediccion = predictor.get_next_day()
        
        resultados = predictor.generar_predicciones_inteligentes(dia_prediccion)
        
        print("\n📋 PREDICCIONES DIVERSIFICADAS Y REALISTAS:")
        print("="*80)
        
        # Analizar diversidad
        numeros_predichos = [int(row['Predicción']) for _, row in resultados.iterrows()]
        diversidad = len(set(numeros_predichos))
        
        print(f"🎯 Diversidad: {diversidad}/6 números únicos")
        
        for _, row in resultados.iterrows():
            print(f"⏰ {row['Hora']}: {row['Predicción']} (Conf: {row['Confianza']}) - {row['Método']}")
            # 🔥 CORREGIDO: Usar 'Números_Alternativos' en lugar de 'Números Alternativas'
            if row['Números_Alternativos']:
                alternativas_str = ", ".join([
                    f"{alt['numero']} ({alt['confianza']})" 
                    for alt in row['Números_Alternativos']
                ])
                print(f"   Alternativas: {alternativas_str}")
        
        # Guardar resultados
        try:
            archivo = f"prediccion_diversificada_{dia_prediccion}.xlsx"
            resultados.to_excel(archivo, index=False)
            print(f"\n💾 Resultados guardados en '{archivo}'")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el archivo: {str(e)}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n¡Predicciones diversificadas completadas! 🎯🍀")

if __name__ == "__main__":
    main_avanzada()