import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import warnings
from collections import Counter

# Configuración
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

class QuinielaPredictorCompacto:
    def __init__(self):
        self.model = None
        self.model_trained = False
        self.report_date = datetime.now().strftime('%Y-%m-%d')
        self.top_numbers = [12, 14, 17, 19, 23, 30, 34, 36]  # Tus números más frecuentes
        self.min_samples = 3  # Reducido para trabajar con pocos datos
        
    def load_data(self, filepath):
        """Carga de datos optimizada para conjuntos pequeños"""
        try:
            df = pd.read_excel(filepath, header=None, usecols=range(8))
            df.columns = ['hora', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
            
            # Limpieza conservadora (mantiene más datos)
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
        
        return pd.DataFrame(records)

    def create_features(self, df):
        """Feature engineering optimizado para pocos datos"""
        # Usar todos los números disponibles (sin filtrado)
        num_counts = Counter(df['numero'])
        self.top_numbers = [num for num, count in num_counts.most_common(8)]
        print(f"\n🔢 Números más frecuentes usados: {self.top_numbers}")
        
        # Características básicas pero efectivas
        df['hora_dia'] = df['hora'] * (df['dia_num'] + 1)
        return df

    def train_model(self, df):
        """Entrenamiento optimizado para pocas muestras"""
        try:
            if len(df) < 10:  # Mínimo muy reducido
                print("\n⚠️ Pocos datos disponibles. Usando modelo simplificado.")
                return self.train_simple_model(df)
            
            X = df[['hora', 'dia_num', 'hora_dia', 'es_finde']]
            y = df['numero']
            
            # Modelo más adecuado para pequeños datasets
            self.model = KNeighborsClassifier(n_neighbors=3)  # Mejor que RF para pocos datos
            self.model.fit(X, y)
            
            # Evaluación con los mismos datos (no hay suficientes para dividir)
            print("\n📊 Rendimiento (usando todos los datos):")
            y_pred = self.model.predict(X)
            print(classification_report(y, y_pred, zero_division=0))
            
            self.model_trained = True
            return self.model
            
        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {str(e)}")
            return None

    def train_simple_model(self, df):
        """Modelo simplificado para muy pocos datos"""
        # Estrategia basada en probabilidades simples
        self.model = "probabilistico"
        self.model_trained = True
        
        # Calcular probabilidades por hora y día
        self.probabilidades = df.groupby(['hora', 'dia_num'])['numero'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()
        
        print("\n🔍 Modelo probabilístico creado basado en frecuencias")
        return self.model

    def predict_for_day(self, dia_semana_str):
        """Predicción adaptada para cada escenario"""
        dias_map = {
            'lunes': 0, 'martes': 1, 'miércoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
        }
        
        try:
            dia_num = dias_map[dia_semana_str.lower()]
        except KeyError:
            raise ValueError(f"Día no válido: {dia_semana_str}")
        
        horas = [10, 11, 12, 13, 14, 15]
        predicciones = []
        
        if not self.model_trained:
            print(f"\n⚠️ Modelo no entrenado. Usando números frecuentes para {dia_semana_str}")
            return self.fallback_predict(dia_semana_str)
        
        if self.model == "probabilistico":
            # Predicción basada en probabilidades simples
            for hora in horas:
                if (hora, dia_num) in self.probabilidades:
                    nums = list(self.probabilidades[(hora, dia_num)].keys())
                    probs = list(self.probabilidades[(hora, dia_num)].values())
                    pred = np.random.choice(nums, p=probs)
                else:
                    pred = np.random.choice(self.top_numbers)
                predicciones.append(f"{pred:02d}")
        else:
            # Predicción con modelo ML
            for hora in horas:
                X_new = pd.DataFrame([{
                    'hora': hora,
                    'dia_num': dia_num,
                    'hora_dia': hora * (dia_num + 1),
                    'es_finde': int(dia_num >= 5)
                }])
                try:
                    pred = self.model.predict(X_new)[0]
                    predicciones.append(f"{pred:02d}")
                except:
                    predicciones.append(f"{np.random.choice(self.top_numbers):02d}")
        
        # Asegurar variedad si hay muchas repeticiones
        if len(set(predicciones)) < 3:
            predicciones = [f"{num:02d}" for num in np.random.choice(self.top_numbers, size=6, replace=True)]
        
        return pd.DataFrame({
            'Hora': [f"{h}:00" for h in horas],
            'Predicción': predicciones,
            'Día': dia_semana_str.capitalize()
        })

    def fallback_predict(self, dia_semana_str):
        """Predicción de respaldo con rotación inteligente"""
        horas = [10, 11, 12, 13, 14, 15]
        # Rotación que asegura variedad
        rotacion = self.top_numbers * 2  # Duplicamos para tener suficientes
        predicciones = [rotacion[i % len(rotacion)] for i in range(len(horas))]
        
        return pd.DataFrame({
            'Hora': [f"{h}:00" for h in horas],
            'Predicción': [f"{p:02d}" for p in predicciones],
            'Día': dia_semana_str.capitalize()
        })

def main():
    print("\n🎲 Quiniela Predictor - Versión para Pocos Datos 🎲")
    print("="*60)
    
    try:
        predictor = QuinielaPredictorCompacto()
        
        print("\n📂 Cargando datos históricos...")
        df_raw = predictor.load_data("historicoquiniela.xlsx")
        
        if df_raw.empty:
            raise ValueError("No se encontraron datos válidos.")
        
        print("\n🔄 Transformando datos...")
        df_long = predictor.transform_to_long_format(df_raw)
        print(f"\n📊 Registros válidos encontrados: {len(df_long)}")
        
        print("\n🔧 Creando características...")
        df_processed = predictor.create_features(df_long)
        
        print("\n🚀 Entrenando modelo optimizado...")
        predictor.train_model(df_processed)
        
        print("\n🔮 Predicciones para la semana:")
        dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        
        resultados = []
        for dia in dias_semana:
            preds = predictor.predict_for_day(dia)
            resultados.append(preds)
            print(f"\n{dia.capitalize()}:")
            print(preds.to_string(index=False))
        
        # Guardar resultados
        pd.concat(resultados).to_excel("predicciones_quiniela_compactas.xlsx", index=False)
        print("\n✅ Proceso completado! Resultados guardados.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("="*60)

if __name__ == "__main__":
    main()