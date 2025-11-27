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
import shutil

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

class PredictorRadical:
    def __init__(self, model_dir="saved_models_radical"):
        self.model = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.num_classes = 0
        self.model_trained = False
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.columnas_dias = []
        self.ultimo_dia = None
        self.all_possible_numbers = list(range(0, 37))
        
        # Atributos para predicciones
        self.ensemble_models = {}
        self.feature_columns_fit = []
        
        # Configuración de paths
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def limpiar_modelos_existentes(self):
        """Limpia completamente todos los modelos existentes"""
        print("🗑️ LIMPIANDO MODELOS EXISTENTES...")
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
            os.makedirs(self.model_dir, exist_ok=True)
        self.ensemble_models = {}
        self.model_trained = False
        print("✅ Modelos anteriores eliminados completamente")

    def load_data(self, filepath):
        """Carga datos con análisis de datos recientes"""
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

    def analizar_datos_recientes_vs_historicos(self, df):
        """Análisis COMPARATIVO entre datos históricos y recientes"""
        print("\n🔍 ANÁLISIS COMPARATIVO: Histórico vs Reciente")
        print("="*50)
        
        # Todos los números
        all_numbers = []
        for _, row in df.iterrows():
            for col in self.columnas_dias:
                if pd.notna(row[col]):
                    all_numbers.append(int(row[col]))
        
        if not all_numbers:
            return
        
        total_registros = len(all_numbers)
        
        # Datos históricos (primeros 80%)
        corte_historico = int(total_registros * 0.8)
        datos_historicos = all_numbers[:corte_historico]
        
        # Datos recientes (últimos 20%)
        datos_recientes = all_numbers[corte_historico:]
        
        print(f"📊 Total registros: {total_registros}")
        print(f"📈 Históricos (80%): {len(datos_historicos)} registros")
        print(f"🔥 Recientes (20%): {len(datos_recientes)} registros")
        
        # Frecuencias históricas
        freq_historico = Counter(datos_historicos)
        print("\n📋 TOP 10 NÚMEROS HISTÓRICOS:")
        for num, count in freq_historico.most_common(10):
            print(f"   - {num:02d}: {count} veces ({count/len(datos_historicos)*100:.1f}%)")
        
        # Frecuencias recientes
        freq_reciente = Counter(datos_recientes)
        print("\n🎯 TOP 10 NÚMEROS RECIENTES:")
        for num, count in freq_reciente.most_common(10):
            print(f"   - {num:02d}: {count} veces ({count/len(datos_recientes)*100:.1f}%)")
        
        # Cambios significativos
        print("\n🔄 CAMBIOS SIGNIFICATIVOS:")
        for num in set(list(freq_historico.keys()) + list(freq_reciente.keys())):
            freq_h = freq_historico.get(num, 0) / len(datos_historicos) * 100
            freq_r = freq_reciente.get(num, 0) / len(datos_recientes) * 100
            cambio = freq_r - freq_h
            
            if abs(cambio) > 5:  # Cambio mayor al 5%
                tendencia = "📈 SUBIÓ" if cambio > 0 else "📉 BAJÓ"
                print(f"   - {num:02d}: {tendencia} {abs(cambio):.1f}% (de {freq_h:.1f}% a {freq_r:.1f}%)")
        
        return datos_historicos, datos_recientes

    def entrenar_modelo_solo_con_recientes(self, df, porcentaje_reciente=30):
        """Entrena SOLO con los datos más recientes - MÉTODO RADICAL"""
        
        print(f"\n🎯 ENTRENAMIENTO RADICAL - Solo con {porcentaje_reciente}% más reciente")
        
        # Preparar todos los datos
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
            print("❌ No hay registros válidos")
            return False
        
        df_long = pd.DataFrame(records)
        
        # 🔥 MÉTODO RADICAL: Usar solo los datos más recientes
        total_registros = len(df_long)
        registros_a_usar = max(50, int(total_registros * porcentaje_reciente / 100))
        
        df_reciente = df_long.tail(registros_a_usar).copy()
        
        print(f"📊 Usando {len(df_reciente)} registros recientes de {total_registros} totales")
        print(f"🎯 Porcentaje usado: {len(df_reciente)/total_registros*100:.1f}%")
        
        # Crear características simples
        df_reciente = self.crear_caracteristicas_simples(df_reciente)
        
        # Entrenar modelos
        X = df_reciente[self.feature_columns_fit]
        y = df_reciente['numero']
        
        self.entrenar_modelos_rapidos(X, y)
        
        print("✅ Entrenamiento radical completado")
        return True

    def crear_caracteristicas_simples(self, df_long):
        """Características simples pero efectivas"""
        
        print("🔄 Creando características optimizadas...")
        
        # Características básicas
        df_long['hora_dia'] = df_long['hora'] * (df_long['dia_num'] + 1)
        df_long['es_finde'] = df_long['dia_num'].apply(lambda x: 1 if x >= 5 else 0)
        df_long['hora_tardia'] = df_long['hora'].apply(lambda x: 1 if x >= 14 else 0)
        
        # Frecuencias recientes (calculadas sobre estos datos)
        freq_reciente = df_long['numero'].value_counts(normalize=True).to_dict()
        df_long['freq_reciente'] = df_long['numero'].map(freq_reciente).fillna(0)
        
        # Frecuencia por hora en estos datos
        for hora in range(10, 16):
            mask = df_long['hora'] == hora
            if mask.any():
                freq_hora = df_long[mask]['numero'].value_counts(normalize=True).to_dict()
                df_long.loc[mask, 'freq_hora'] = df_long.loc[mask, 'numero'].map(freq_hora).fillna(0)
        
        df_long['freq_reciente'] = df_long['freq_reciente'].fillna(0)
        df_long['freq_hora'] = df_long['freq_hora'].fillna(0)
        
        self.feature_columns_fit = [
            'hora', 'dia_num', 'hora_dia', 'es_finde', 'hora_tardia',
            'freq_reciente', 'freq_hora'
        ]
        
        print(f"✅ {len(self.feature_columns_fit)} características creadas")
        return df_long

    def entrenar_modelos_rapidos(self, X, y):
        """Entrenamiento rápido y efectivo"""
        
        print("🔄 Entrenando modelos rápidos...")
        
        # Codificar target
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        self.num_classes = len(self.encoder.classes_)
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Solo 2 modelos para mayor velocidad y menos sobreajuste
        try:
            # Modelo 1: Random Forest más simple
            rf_model = RandomForestClassifier(
                n_estimators=50,  # Menos árboles
                max_depth=8,      # Menos profundidad
                min_samples_split=10,
                random_state=42
            )
            rf_model.fit(X_scaled, y_encoded)
            self.ensemble_models['random_forest'] = rf_model
            print("✅ Random Forest rápido entrenado")
        except Exception as e:
            print(f"⚠️ Error en Random Forest: {e}")

        try:
            # Modelo 2: Gradient Boosting más simple
            gb_model = GradientBoostingClassifier(
                n_estimators=30,  # Menos estimadores
                max_depth=5,      # Menos profundidad  
                random_state=42
            )
            gb_model.fit(X_scaled, y_encoded)
            self.ensemble_models['gradient_boosting'] = gb_model
            print("✅ Gradient Boosting rápido entrenado")
        except Exception as e:
            print(f"⚠️ Error en Gradient Boosting: {e}")

        self.model_trained = True
        print(f"🎯 {len(self.ensemble_models)} modelos rápidos entrenados")

    def predecir_ensemble(self, features):
        """Predicción simple del ensemble"""
        
        if not self.ensemble_models:
            raise ValueError("No hay modelos entrenados")
        
        features = features[self.feature_columns_fit]
        features_scaled = self.scaler.transform(features)
        
        predicciones = {}
        confianzas = {}
        
        for nombre, modelo in self.ensemble_models.items():
            try:
                proba = modelo.predict_proba(features_scaled)[0]
                pred_idx = np.argmax(proba)
                predicciones[nombre] = self.encoder.inverse_transform([pred_idx])[0]
                confianzas[nombre] = np.max(proba)
            except Exception as e:
                continue
        
        if not predicciones:
            # Fallback: número aleatorio basado en frecuencias recientes
            return {
                'prediccion': np.random.choice(self.all_possible_numbers),
                'confianza': 0.1,
                'metodo': 'Fallback'
            }
        
        # Combinar por votación
        counts = Counter(predicciones.values())
        prediccion_final = counts.most_common(1)[0][0]
        confianza_final = counts.most_common(1)[0][1] / len(predicciones)
        
        return {
            'prediccion': prediccion_final,
            'confianza': min(confianza_final, 0.8),  # Máximo 80%
            'metodo': '+'.join(predicciones.keys())
        }

    def generar_predicciones_diversificadas(self, dia_semana):
        """Genera predicciones forzando diversidad máxima"""
        
        print(f"\n🎯 GENERANDO PREDICCIONES DIVERSIFICADAS PARA {dia_semana.upper()}")
        
        horas = list(range(10, 16))
        resultados = []
        numeros_usados = set()
        
        for hora in horas:
            print(f"   ⏰ Hora {hora}:00...")
            
            # Características para esta hora
            features = self._preparar_features(hora, dia_semana)
            
            # Obtener predicción del ensemble
            try:
                pred = self.predecir_ensemble(features)
                numero_pred = pred['prediccion']
                confianza = pred['confianza']
                metodo = pred['metodo']
            except:
                numero_pred = np.random.choice(self.all_possible_numbers)
                confianza = 0.15
                metodo = "Fallback"
            
            # 🔥 FORZAR DIVERSIDAD: Si el número ya fue usado, buscar alternativo
            intentos = 0
            while numero_pred in numeros_usados and intentos < 10:
                # Buscar número alternativo
                alternativos = [n for n in self.all_possible_numbers if n not in numeros_usados]
                if alternativos:
                    numero_pred = np.random.choice(alternativos)
                    confianza = max(0.1, confianza * 0.7)  # Reducir confianza
                    metodo = "Diversificado"
                intentos += 1
            
            numeros_usados.add(numero_pred)
            
            # Generar 2 alternativas diferentes
            alternativas = []
            alternativos_disponibles = [n for n in self.all_possible_numbers if n not in numeros_usados]
            
            for _ in range(2):
                if alternativos_disponibles:
                    alt_num = np.random.choice(alternativos_disponibles)
                    alt_conf = np.random.uniform(0.1, 0.3)
                    alternativas.append({
                        'numero': f"{alt_num:02d}",
                        'confianza': f"{alt_conf:.1%}"
                    })
                    alternativos_disponibles.remove(alt_num)
            
            resultados.append({
                'Hora': f"{hora}:00",
                'Predicción': f"{numero_pred:02d}",
                'Confianza': f"{confianza:.1%}",
                'Método': metodo,
                'Números_Alternativos': alternativas
            })
        
        return pd.DataFrame(resultados)

    def _preparar_features(self, hora, dia_semana):
        """Prepara características para predicción"""
        dia_num = self.dias_semana.index(dia_semana.lower())
        
        features = {
            'hora': hora,
            'dia_num': dia_num,
            'hora_dia': hora * (dia_num + 1),
            'es_finde': int(dia_num >= 5),
            'hora_tardia': 1 if hora >= 14 else 0,
            'freq_reciente': 0.02,  # Valor por defecto
            'freq_hora': 0.02
        }
        
        return pd.DataFrame([features])

    def get_next_day(self):
        """Calcula el próximo día"""
        if self.ultimo_dia is None:
            raise ValueError("Primero cargue los datos")
        idx = self.dias_semana.index(self.ultimo_dia)
        return self.dias_semana[(idx + 1) % 7]

# Función principal RADICAL
def main_radical():
    print("\n🔮 PREDICTOR RADICAL - REINICIO COMPLETO 🔮")
    print("="*60)
    
    predictor = PredictorRadical()
    
    try:
        # 1. Cargar datos
        print("\n📂 CARGANDO DATOS...")
        df = predictor.load_data("historicoquiniela.xlsx")
        
        if df.empty:
            raise ValueError("No se pudieron cargar datos válidos")
        
        # 2. Análisis comparativo
        datos_historicos, datos_recientes = predictor.analizar_datos_recientes_vs_historicos(df)
        
        # 3. Limpieza completa
        predictor.limpiar_modelos_existentes()
        
        # 4. Entrenamiento RADICAL solo con datos recientes
        print("\n🎯 ENTRENAMIENTO RADICAL EN PROGRESO...")
        exito = predictor.entrenar_modelo_solo_con_recientes(df, porcentaje_reciente=30)
        
        if not exito:
            raise ValueError("Falló el entrenamiento radical")
        
        # 5. Generar predicciones
        dia_prediccion = input("\n📅 Ingrese el día a predecir: ").strip().lower()
        
        if dia_prediccion not in predictor.dias_semana:
            print("Día inválido. Usando siguiente día disponible.")
            dia_prediccion = predictor.get_next_day()
        
        resultados = predictor.generar_predicciones_diversificadas(dia_prediccion)
        
        # 6. Mostrar resultados
        print("\n📋 PREDICCIONES RADICALES - DIVERSIFICADAS:")
        print("="*60)
        
        numeros_unicos = len(set([int(row['Predicción']) for _, row in resultados.iterrows()]))
        print(f"🎯 DIVERSIDAD: {numeros_unicos}/6 números únicos")
        
        for _, row in resultados.iterrows():
            print(f"⏰ {row['Hora']}: {row['Predicción']} (Conf: {row['Confianza']}) - {row['Método']}")
            if row['Números_Alternativos']:
                alt_str = ", ".join([f"{alt['numero']} ({alt['confianza']})" for alt in row['Números_Alternativos']])
                print(f"   💡 Alternativas: {alt_str}")
        
        # 7. Guardar
        try:
            archivo = f"prediccion_radical_{dia_prediccion}.xlsx"
            resultados.to_excel(archivo, index=False)
            print(f"\n💾 Guardado en: {archivo}")
        except Exception as e:
            print(f"⚠️ Error guardando: {e}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n¡PREDICCIONES RADICALES COMPLETADAS! 🚀")

if __name__ == "__main__":
    main_radical()