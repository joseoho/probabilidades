import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import warnings
from collections import Counter
from imblearn.over_sampling import SMOTE

# Configuración
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

class QuinielaPredictorOptimizado:
    def __init__(self):
        self.model = None
        self.model_trained = False
        self.df_processed = None
        self.top_numbers = [12, 14, 17, 19, 23, 30, 34, 36]
        self.min_samples = 3
        self.n_neighbors = 5  # Inicialmente 5, pero se ajustará dinámicamente
        
    def load_data(self, filepath):
        """Carga de datos optimizada para conjuntos pequeños"""
        try:
            df = pd.read_excel(filepath, header=None, usecols=range(8))
            df.columns = ['hora', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(0, np.nan)
            
            df = df.dropna()
            return df
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            return pd.DataFrame()

    def transform_to_long_format(self, df):
        """Transformación que conserva la máxima información"""
        records = []
        for _, row in df.iterrows():
            for dia in ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']:
                num = row[dia]
                if pd.notnull(num) and 1 <= num <= 36:
                    records.append({
                        'dia_semana': dia,
                        'hora': int(row['hora']),
                        'numero': int(num),
                        'dia_num': ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo'].index(dia),
                        'es_finde': int(dia in ['sábado', 'domingo'])
                    })
        
        df_long = pd.DataFrame(records)
        df_long['numero_anterior'] = df_long.groupby('hora')['numero'].shift(1).fillna(0)  # Nuevo feature
        return df_long

    def create_features(self, df):
        """Feature engineering optimizado para pocos datos"""
        num_counts = Counter(df['numero'])
        self.top_numbers = [num for num, count in num_counts.most_common(8)]
        print(f"\n🔢 Números más frecuentes usados: {self.top_numbers}")
        
        # Agrupamos los números poco frecuentes
        df['numero_grupo'] = df['numero'].apply(lambda x: 99 if num_counts[x] < self.min_samples else x)
        
        df['hora_dia'] = df['hora'] * (df['dia_num'] + 1)
        return df

    def train_model(self, df):
        """Entrenamiento optimizado para pocas muestras"""
        try:
            self.df_processed = df

            X = df[['hora', 'dia_num', 'hora_dia', 'es_finde', 'numero_anterior']]
            y = df['numero_grupo']

            # Ajuste dinámico de `n_neighbors`
            if len(X) < self.n_neighbors:
                self.n_neighbors = max(1, len(X) - 1)  # Ajustamos a un valor válido
                print(f"\n⚠️ Ajustando n_neighbors a {self.n_neighbors} debido al tamaño del dataset.")

            # Validamos si hay suficientes datos
            if len(X) < 6:
                print("\n⚠️ Pocos datos disponibles. Usando modelo probabilístico de respaldo.")
                return self.train_simple_model(df)

            # Aplicamos SMOTE para mejorar el balance de clases
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Modelo optimizado
            self.model = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, verbose=False)
            self.model.fit(X_resampled, y_resampled)

            # Evaluación
            print("\n📊 Evaluación del modelo:")
            y_pred = self.model.predict(X)
            print(classification_report(y, y_pred, zero_division=0))

            self.model_trained = True
            return self.model
            
        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {str(e)}")
            return None

    def train_simple_model(self, df):
        """Modelo probabilístico cuando hay pocos datos"""
        self.model = "probabilistico"
        self.model_trained = True

        self.probabilidades = df.groupby(['hora', 'dia_num'])['numero'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()

        print("\n🔍 Modelo probabilístico creado basado en frecuencias")
        return self.model

    def predict_for_day(self, dia_semana_str):
        """Genera predicciones con verificación de modelo"""
        dias_map = {
            'lunes': 0, 'martes': 1, 'miércoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
        }
        
        dia_num = dias_map.get(dia_semana_str.lower())
        if dia_num is None:
            raise ValueError(f"Día no válido: {dia_semana_str}")
        
        horas = [10, 11, 12, 13, 14, 15]
        predicciones = []

        if not self.model_trained or self.model is None:
            print(f"\n⚠️ Modelo no entrenado. Usando números frecuentes para {dia_semana_str}")
            return self.fallback_predict(dia_semana_str)
        
        for hora in horas:
            X_new = pd.DataFrame([{
                'hora': hora,
                'dia_num': dia_num,
                'hora_dia': hora * (dia_num + 1),
                'es_finde': int(dia_num >= 5),
                'numero_anterior': np.random.choice(self.top_numbers)
            }])
            pred = self.model.predict(X_new)[0]
            predicciones.append(f"{pred:02d}")
        
        return pd.DataFrame({'Hora': [f"{h}:00" for h in horas], 'Predicción': predicciones, 'Día': dia_semana_str.capitalize()})

    def fallback_predict(self, dia_semana_str):
        """Predicción de respaldo usando números frecuentes"""
        horas = [10, 11, 12, 13, 14, 15]
        predicciones = [np.random.choice(self.top_numbers) for _ in horas]
        
        return pd.DataFrame({'Hora': [f"{h}:00" for h in horas], 'Predicción': predicciones, 'Día': dia_semana_str.capitalize()})

def main():
    print("\n🎲 Quiniela Predictor - Versión Mejorada 🎲")
    
    predictor = QuinielaPredictorOptimizado()
    df_raw = predictor.load_data("historicoquiniela.xlsx")
    df_long = predictor.transform_to_long_format(df_raw)
    df_processed = predictor.create_features(df_long)
    predictor.train_model(df_processed)

    dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    resultados = pd.concat([predictor.predict_for_day(dia) for dia in dias_semana])

    resultados.to_excel("predicciones_quiniela_mejorado.xlsx", index=False)

if __name__ == "__main__":
    main()
