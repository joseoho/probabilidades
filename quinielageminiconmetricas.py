import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import warnings
from collections import Counter
import joblib

# Configuración global
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

class QuinielaPredictor:
    def __init__(self):
        self.model = None
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

    def load_data(self, filepath):
        """Carga datos con manejo robusto de encabezados"""
        try:
            # Leer el archivo ignorando encabezados existentes
            df = pd.read_excel(filepath, header=None)
            
            # Verificar estructura básica
            if df.shape[1] < 8:
                raise ValueError("El archivo debe tener al menos 8 columnas (hora + 7 días)")
            
            # Asignar nombres de columnas
            df.columns = ['hora'] + [f'dia_{i}' for i in range(1, df.shape[1])]
            self.columnas_dias = df.columns[1:]  # Todas excepto 'hora'
            
            print(f"DEBUG: Columnas asignadas - Hora: 'hora', Días: {self.columnas_dias[:7]}... (total: {len(self.columnas_dias)})")

            # Detectar el último día con datos (asignando a días de la semana cíclicamente)
            for i in reversed(range(len(self.columnas_dias))):
                if not df.iloc[:, i+1].isnull().all():  # +1 para saltar columna hora
                    self.ultimo_dia = self.dias_semana[i % 7]
                    break
            
            if self.ultimo_dia is None:
                raise ValueError("No se encontraron datos válidos en ninguna columna")
            
            # Limpieza de datos
            for col in self.columnas_dias:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: x if pd.isna(x) or (0 <= x <= 36) else np.nan)
            
            # Eliminar filas completamente vacías en los días
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
            
        # Aplanar todos los números
        all_nums = []
        for col in self.columnas_dias:
            all_nums.extend(df[col].dropna().astype(int).tolist())
        
        if not all_nums:
            print("No se encontraron números válidos")
            return
            
        # Frecuencias generales
        counts = Counter(all_nums)
        self.most_frequent_overall = [num for num, _ in counts.most_common(8)]
        
        # Números recientes (últimas 100 filas)
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
        
        # Calcular probabilidades por hora y día
        self.probabilidades = df_long.groupby(['hora', 'dia_num'])['numero'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()
        
        print("Modelo probabilístico entrenado (basado en frecuencias)")
        self.metrics_results = {"Modelo": "Probabilístico (frecuencias)"}
        return self.model

    def train_model(self, df):
        """Entrena el modelo adaptativo"""
        # Convertir a formato largo
        records = []
        for _, row in df.iterrows():
            for i, col in enumerate(self.columnas_dias):
                num = row[col]
                if pd.notna(num):
                    dia_num = i % 7  # Asignar días cíclicamente
                    records.append({
                        'hora': int(row['hora']),
                        'numero': int(num),
                        'dia_num': dia_num,
                        'es_finde': int(dia_num >= 5)  # 1 si es fin de semana
                    })
        
        if not records:
            print("No hay datos para entrenar")
            return self.train_probabilistic_model(pd.DataFrame())
            
        df_long = pd.DataFrame(records)
        
        # Crear características adicionales
        df_long['hora_dia'] = df_long['hora'] * (df_long['dia_num'] + 1)
        
        # Entrenamiento según cantidad de datos
        if len(df_long) < 50:
            print("Pocos datos. Usando modelo probabilístico")
            return self.train_probabilistic_model(df_long)
            
        try:
            X = df_long[['hora', 'dia_num', 'hora_dia', 'es_finde']]
            y = df_long['numero']
            
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
            self.model_trained = True
            self.y_pred_test = self.model.predict(self.X_test)
            
            self.evaluate_model()
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
            if strategy == "modelo" and self.model_trained and isinstance(self.model, RandomForestClassifier):
                try:
                    X_new = pd.DataFrame([{
                        'hora': hora,
                        'dia_num': dia_num,
                        'hora_dia': hora * (dia_num + 1),
                        'es_finde': int(dia_num >= 5)
                    }])
                    proba = self.model.predict_proba(X_new)[0]
                    nums = self.model.classes_
                    num = np.random.choice(nums, p=proba)
                except:
                    num = np.random.choice(self.all_possible_numbers)
                    
            elif strategy == "calientes":
                num = np.random.choice(self.hot_numbers if self.hot_numbers else self.all_possible_numbers)
                
            elif strategy == "frios":
                num = np.random.choice(self.cold_numbers if self.cold_numbers else self.all_possible_numbers)
                
            elif strategy == "balanceado":
                combined = list(set(self.hot_numbers + self.cold_numbers + self.most_frequent_overall))
                num = np.random.choice(combined if combined else self.all_possible_numbers)
                
            else:  # Aleatorio
                num = np.random.choice(self.all_possible_numbers)
                
            predicciones.append(f"{num:02d}")
            
        # Mezclar predicciones para evitar repeticiones
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

def main():
    print("\n🔮 Quiniela Predictor - Versión Corregida 🔮")
    print("="*50)
    
    predictor = QuinielaPredictor()
    
    try:
        print("\n📂 Cargando datos...")
        df = predictor.load_data("historicoquiniela.xlsx")
        
        if df.empty:
            raise ValueError("No se pudieron cargar datos válidos")
            
        print(f"\n📆 Último día con datos: {predictor.ultimo_dia.capitalize()}")
        siguiente_dia = predictor.get_next_day()
        
        # Interfaz de usuario simple
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
        
        print("\nEstrategias disponibles:")
        print("a. Modelo predictivo")
        print("b. Números calientes")
        print("c. Números fríos")
        print("d. Balanceado")
        print("e. Aleatorio")
        estrategia = input("Seleccione estrategia (a-e): ").lower().strip()
        
        estrategias = {
            'a': 'modelo',
            'b': 'calientes',
            'c': 'frios',
            'd': 'balanceado',
            'e': 'aleatorio'
        }
        estrategia = estrategias.get(estrategia, 'aleatorio')
        
        # Procesamiento
        print("\n🔧 Calculando estadísticas...")
        predictor.calculate_number_stats(df)
        
        print("\n🤖 Entrenando modelo...")
        predictor.train_model(df)
        
        print(f"\n🎯 Generando predicciones para {dia.capitalize()}...")
        resultados = predictor.predict_day(dia, estrategia)
        
        print("\n📋 Resultados:")
        print(resultados.to_string(index=False))
        
        # Guardar resultados
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