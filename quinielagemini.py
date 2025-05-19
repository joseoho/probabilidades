import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# Configuración
plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)

class QuinielaPredictor:
    def __init__(self):
        self.model = None
        self.class_weights = None

    def load_data(self, filepath):
        """Carga y transforma los datos"""
        try:
            df = pd.read_excel(filepath, header=None)
            df = df.iloc[:, :8]
            df.columns = ['hora', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            return df
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            return pd.DataFrame()

    def transform_to_long_format(self, df, start_date='2023-01-02'):
        """Convierte a formato largo con fechas generadas"""
        records = []
        num_semanas = len(df) // 6
        fechas = pd.date_range(start=start_date, periods=num_semanas*7, freq='D')

        for semana in range(num_semanas):
            for i, dia in enumerate(['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']):
                fecha = fechas[semana*7 + i]
                for _, row in df.iterrows():
                    records.append({
                        'fecha': fecha,
                        'dia_semana': dia,
                        'hora': int(row['hora']),
                        'numero': int(row[dia]) if row[dia] != '00' else 0
                    })
        return pd.DataFrame(records)

    def create_features(self, df):
        """Crea características y agrupa números poco frecuentes"""
        counts = df['numero'].value_counts()
        rare_numbers = counts[counts < 3].index
        df['numero_grupo'] = df['numero'].apply(lambda x: 37 if x in rare_numbers else x)

        # Características temporales
        df['dia_num'] = df['dia_semana'].map({
            'lunes': 0, 'martes': 1, 'miércoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
        })
        df['hora_dia'] = df['hora'] * (df['dia_num'] + 1)
        df['es_finde'] = (df['dia_num'] >= 5).astype(int)
        df['mes'] = df['fecha'].dt.month  # Añadimos el mes como característica
        return df

    def train_model(self, df):
        """Entrena el modelo con validación cruzada de series de tiempo"""
        try:
            X = df[['hora', 'dia_num', 'hora_dia', 'es_finde', 'mes']]  # Incluimos 'mes'
            y = df['numero_grupo']

            # Calcular pesos de clases
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, weights))

            # Modelo Gradient Boosting
            self.model = GradientBoostingClassifier(
                n_estimators=100,  # Ajusta según sea necesario
                max_depth=3,       # Ajusta según sea necesario
                learning_rate=0.1, # Ajusta según sea necesario
                random_state=42
            )

            # Validación cruzada de series de tiempo
            tscv = TimeSeriesSplit(n_splits=5)  # 5 splits es un buen punto de partida
            
            # Grid Search para ajustar hiperparámetros (opcional, pero recomendado)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=tscv, scoring='f1_macro', n_jobs=-1) #Usamos f1_macro
            grid_search.fit(X, y)
            
            print("Mejores hiperparámetros:", grid_search.best_params_)
            self.model = grid_search.best_estimator_
            
            # Reporte de clasificación con el mejor modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Usamos un test del 20%
            print(classification_report(y_test, self.model.predict(X_test), zero_division=0))

            return self.model
        except Exception as e:
            print(f"Error entrenando modelo: {str(e)}")
            return None

    def predict_for_day(self, dia_semana_str):
        """Genera predicciones para un día específico"""
        if not self.model:
            raise ValueError("Modelo no entrenado")

        dias_map = {
            'lunes': 0, 'martes': 1, 'miércoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
        }
        dia_num = dias_map[dia_semana_str.lower()]

        horas = [10, 11, 12, 13, 14, 15]
        predicciones = []

        fecha_prediccion = datetime.now()  # Usamos la fecha actual para el mes
        mes_prediccion = fecha_prediccion.month
        
        for hora in horas:
            X_new = pd.DataFrame([{
                'hora': hora,
                'dia_num': dia_num,
                'hora_dia': hora * (dia_num + 1),
                'es_finde': int(dia_num >= 5),
                'mes': mes_prediccion
            }])
            pred = self.model.predict(X_new)[0]
            if pred == 37:
                pred = np.random.choice([14, 30, 34])
            predicciones.append(f"{pred:02d}")
        return predicciones

def main():
    print("🔮 Sistema de Predicción Mejorado 🔮")

    try:
        predictor = QuinielaPredictor()

        print("\n📂 Cargando datos...")
        df_raw = predictor.load_data("historicoquiniela.xlsx")

        print("\n🔄 Transformando datos...")
        df_long = predictor.transform_to_long_format(df_raw)

        print("\n🔧 Creando características...")
        df_processed = predictor.create_features(df_long)

        print("\n📊 Distribución de números:")
        print(df_processed['numero_grupo'].value_counts().sort_index())

        print("\n🚀 Entrenando modelo...")
        predictor.train_model(df_processed)

        print("\n🔮 Predicciones para el próximo lunes:")
        preds = predictor.predict_for_day('lunes')

        resultados = pd.DataFrame({
            'Hora': [f"{h}:00" for h in [10, 11, 12, 13, 14, 15]],
            'Predicción': preds
        })

        resultados.to_excel("predicciones_quiniela.xlsx", index=False)
        joblib.dump(predictor, 'modelo_quiniela.pkl')

        print("\n✅ Resultados:")
        print(resultados.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
