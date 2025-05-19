import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

class QuinielaPredictor:
    def __init__(self):
        self.model = None
        self.class_weights = None
        self.df_processed = None  # Guardamos el dataframe para acceder en toda la clase

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
        fechas = pd.date_range(start=start_date, periods=len(df) * 7, freq='D')

        for i, dia in enumerate(['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']):
            for _, row in df.iterrows():
                records.append({
                    'fecha': fechas[i],
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

        # Conversión correcta de clases en `y`
        df['numero_grupo'] = df['numero_grupo'].astype('category').cat.codes

        # Características temporales
        df['dia_num'] = df['dia_semana'].map({
            'lunes': 0, 'martes': 1, 'miércoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6
        })
        df['hora_dia'] = df['hora'] * (df['dia_num'] + 1)
        df['es_finde'] = (df['dia_num'] >= 5).astype(int)
        df['mes'] = df['fecha'].dt.month
        
        # Tendencias históricas
        df['frecuencia_numero'] = df.groupby('numero')['numero'].transform('count')
        df['media_hora'] = df.groupby('hora')['numero'].transform('mean')

        return df

    def train_model(self, df):
        """Entrena el modelo con validación cruzada de series de tiempo"""
        try:
            self.df_processed = df  # Guardamos el dataframe procesado para usarlo en otras funciones

            X = df[['hora', 'dia_num', 'hora_dia', 'es_finde', 'mes', 'frecuencia_numero', 'media_hora']]
            y = df['numero_grupo']

            # Calcular pesos de clases
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, weights))

            # Modelo XGBoost
            self.model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=3,
                learning_rate=0.1,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )

            # Validación cruzada
            tscv = TimeSeriesSplit(n_splits=5)
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
            search = RandomizedSearchCV(self.model, param_dist, cv=tscv, scoring='accuracy', n_jobs=-1)
            search.fit(X, y)
            
            print("Mejores hiperparámetros:", search.best_params_)
            self.model = search.best_estimator_

            # Evaluación del modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = self.model.predict(X_test)

            print("\n🔹 Reporte de Clasificación")
            print(classification_report(y_test, y_pred, zero_division=0))
            print("\n🔹 Precisión:", accuracy_score(y_test, y_pred))
            print("\n🔹 ROC AUC:", roc_auc_score(y_test, self.model.predict_proba(X_test), multi_class='ovr'))

            return self.model
        except Exception as e:
            print(f"Error entrenando modelo: {str(e)}")
            return None

    def predict_for_day(self, dia_semana_str):
        """Genera predicciones para un día específico"""
        if not self.model:
            raise ValueError("Modelo no entrenado")

        dias_map = { 'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6 }
        dia_num = dias_map[dia_semana_str.lower()]

        horas = [10, 11, 12, 13, 14, 15]
        predicciones = []

        fecha_prediccion = datetime.now()
        mes_prediccion = fecha_prediccion.month
        
        for hora in horas:
            X_new = pd.DataFrame([{
                'hora': hora,
                'dia_num': dia_num,
                'hora_dia': hora * (dia_num + 1),
                'es_finde': int(dia_num >= 5),
                'mes': mes_prediccion,
                'frecuencia_numero': np.random.choice(self.df_processed['frecuencia_numero']),
                'media_hora': self.df_processed[self.df_processed['hora'] == hora]['media_hora'].mean()
            }])
            pred = self.model.predict(X_new)[0]
            predicciones.append(f"{pred:02d}")
        return predicciones

def main():
    print("🔮 Sistema de Predicción Mejorado 🔮")

    try:
        predictor = QuinielaPredictor()

        df_raw = predictor.load_data("historicoquiniela.xlsx")
        df_long = predictor.transform_to_long_format(df_raw)
        df_processed = predictor.create_features(df_long)
        predictor.train_model(df_processed)

        preds = predictor.predict_for_day('lunes')
        resultados = pd.DataFrame({'Hora': [f"{h}:00" for h in [10, 11, 12, 13, 14, 15]], 'Predicción': preds})

        resultados.to_excel("predicciones_quiniela.xlsx", index=False)
        joblib.dump(predictor, 'modelo_quiniela.pkl')

        print("\n✅ Resultados:\n", resultados.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
