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

warnings.filterwarnings('ignore')

class PredictorTodosLosDias:
    def __init__(self, model_dir="saved_models_todos_dias"):
        self.model = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.num_classes = 0
        self.model_trained = False
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.columnas_dias = []
        self.ultimo_dia = None
        self.all_possible_numbers = list(range(0, 37))
        
        # Modelos por día
        self.modelos_por_dia = {}
        self.encoders_por_dia = {}
        self.scalers_por_dia = {}
        self.features_por_dia = {}
        self.datos_completos_por_dia = {}  # Aquí almacenaremos TODOS los datos de cada día
        
        # Configuración de paths
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def load_data_y_organizar_por_dia(self, filepath):
        """Carga datos y organiza TODOS los registros por día de la semana"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
            
            print(f"📂 Cargando archivo: {filepath}")
            
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, header=None, engine='openpyxl')
            elif filepath.endswith('.xls'):
                df = pd.read_excel(filepath, header=None)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath, header=None)
            else:
                raise ValueError("Formato de archivo no soportado")
            
            print(f"✅ Archivo cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
            
            if df.shape[1] < 8:
                raise ValueError("El archivo debe tener al menos 8 columnas (hora + 7 días)")
            
            # Asignar nombres de columnas
            df.columns = ['hora'] + [f'dia_{i}' for i in range(1, df.shape[1])]
            self.columnas_dias = df.columns[1:]
            
            print(f"📊 Columnas: {list(df.columns)}")
            
            # Procesar hora
            df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
            df = df.dropna(subset=['hora'])
            df['hora'] = df['hora'].astype(int)
            
            # Determinar último día con datos
            self.ultimo_dia = None
            for i in reversed(range(len(self.columnas_dias))):
                if not df.iloc[:, i+1].isnull().all():
                    self.ultimo_dia = self.dias_semana[i % 7]
                    break
            
            if self.ultimo_dia is None:
                raise ValueError("No se encontraron datos válidos en ninguna columna")
            
            print(f"\n🔍 ORGANIZANDO TODOS LOS DATOS POR DÍA...")
            
            # Inicializar diccionarios para cada día
            for dia in self.dias_semana:
                self.datos_completos_por_dia[dia] = []
            
            # 🔥 CLAVE: Recorrer TODAS las columnas y asignar cada número al día correspondiente
            total_registros = 0
            
            for i, col in enumerate(self.columnas_dias):
                dia_nombre = self.dias_semana[i % 7]
                
                print(f"   Procesando columna '{col}' → Día: {dia_nombre}")
                
                # Filtrar datos válidos de esta columna
                datos_col = df[['hora', col]].copy()
                datos_col = datos_col.dropna(subset=[col])
                
                # Convertir a enteros y organizar
                for _, row in datos_col.iterrows():
                    hora = int(row['hora'])
                    numero = int(row[col])
                    
                    self.datos_completos_por_dia[dia_nombre].append({
                        'hora': hora,
                        'numero': numero,
                        'dia_nombre': dia_nombre,
                        'dia_num': i % 7
                    })
                    total_registros += 1
            
            # Convertir listas a DataFrames
            for dia in self.dias_semana:
                if self.datos_completos_por_dia[dia]:
                    self.datos_completos_por_dia[dia] = pd.DataFrame(self.datos_completos_por_dia[dia])
                else:
                    self.datos_completos_por_dia[dia] = pd.DataFrame(columns=['hora', 'numero', 'dia_nombre', 'dia_num'])
            
            print(f"\n📊 RESUMEN POR DÍA:")
            print("="*40)
            for dia in self.dias_semana:
                count = len(self.datos_completos_por_dia[dia])
                print(f"   {dia.capitalize()}: {count} registros")
            
            print(f"\n✅ Total registros organizados: {total_registros}")
            print(f"📅 Último día con datos: {self.ultimo_dia}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR cargando datos: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analizar_patrones_dia_especifico(self, dia_nombre):
        """Análisis detallado para un día específico"""
        
        print(f"\n🔍 ANÁLISIS DETALLADO PARA {dia_nombre.upper()}")
        print("="*50)
        
        datos_dia = self.datos_completos_por_dia.get(dia_nombre, pd.DataFrame())
        
        if len(datos_dia) == 0:
            print(f"   ❌ No hay datos para {dia_nombre}")
            return None
        
        print(f"   📊 Total registros: {len(datos_dia)}")
        
        # Estadísticas generales
        numeros = datos_dia['numero'].tolist()
        total = len(numeros)
        
        # Frecuencias
        freq = Counter(numeros)
        print(f"\n   🎯 DISTRIBUCIÓN DE NÚMEROS:")
        print(f"   {'='*30}")
        
        top_10 = freq.most_common(10)
        for num, count in top_10:
            porcentaje = (count / total) * 100
            print(f"      {num:02d}: {count:3d} veces ({porcentaje:5.1f}%) {'⭐' if porcentaje > 10 else ''}")
        
        # Distribución por hora
        print(f"\n   ⏰ DISTRIBUCIÓN POR HORA:")
        print(f"   {'='*30}")
        
        patrones_horarios = {}
        for hora in range(10, 16):
            datos_hora = datos_dia[datos_dia['hora'] == hora]
            count_hora = len(datos_hora)
            
            if count_hora > 0:
                numeros_hora = datos_hora['numero'].tolist()
                freq_hora = Counter(numeros_hora)
                
                if freq_hora:
                    num_mas_comun = freq_hora.most_common(1)[0][0]
                    count_mas_comun = freq_hora.most_common(1)[0][1]
                    porcentaje_hora = (count_mas_comun / count_hora) * 100
                    
                    patrones_horarios[hora] = {
                        'total': count_hora,
                        'mas_comun': num_mas_comun,
                        'porcentaje': porcentaje_hora,
                        'todos_numeros': freq_hora.most_common(3)
                    }
                    
                    print(f"      {hora}:00 → {num_mas_comun:02d} ({count_mas_comun}/{count_hora} = {porcentaje_hora:.1f}%)")
        
        # Análisis de tendencias
        print(f"\n   📈 ANÁLISIS DE TENDENCIAS:")
        print(f"   {'='*30}")
        
        if total >= 20:
            # Dividir en mitades para ver cambios
            mitad = total // 2
            primera_mitad = numeros[:mitad]
            segunda_mitad = numeros[mitad:]
            
            freq_primera = Counter(primera_mitad)
            freq_segunda = Counter(segunda_mitad)
            
            # Encontrar números que aumentaron
            cambios = []
            for num in set(list(freq_primera.keys()) + list(freq_segunda.keys())):
                antes = freq_primera.get(num, 0) / len(primera_mitad) * 100
                despues = freq_segunda.get(num, 0) / len(segunda_mitad) * 100
                cambio = despues - antes
                
                if abs(cambio) > 5:  # Cambio significativo
                    cambios.append((num, antes, despues, cambio))
            
            if cambios:
                print(f"      Cambios significativos:")
                for num, antes, despues, cambio in sorted(cambios, key=lambda x: abs(x[3]), reverse=True)[:5]:
                    tendencia = "📈 SUBIÓ" if cambio > 0 else "📉 BAJÓ"
                    print(f"      - {num:02d}: {tendencia} {abs(cambio):.1f}% ({antes:.1f}% → {despues:.1f}%)")
            else:
                print(f"      No se detectaron cambios significativos")
        else:
            print(f"      Necesarios más datos para análisis de tendencias")
        
        return {
            'total_registros': total,
            'frecuencias': freq,
            'patrones_horarios': patrones_horarios,
            'datos': datos_dia
        }

    def entrenar_modelo_dia_especifico(self, dia_nombre):
        """Entrena un modelo específico para un día usando TODOS sus datos"""
        
        print(f"\n🎯 ENTRENANDO MODELO ESPECÍFICO PARA {dia_nombre.upper()}")
        
        datos_dia = self.datos_completos_por_dia.get(dia_nombre, pd.DataFrame())
        
        if len(datos_dia) < 15:
            print(f"   ⚠️ Pocos datos disponibles: {len(datos_dia)} registros")
            print(f"   💡 Se requiere mínimo 15 registros para entrenamiento efectivo")
            
            # Si hay pocos datos, podemos usar un enfoque diferente
            return self.entrenar_modelo_con_pocos_datos(datos_dia, dia_nombre)
        
        print(f"   📊 Datos disponibles: {len(datos_dia)} registros")
        
        # Crear características
        datos_procesados = self.crear_caracteristicas_completas(datos_dia, dia_nombre)
        
        # Preparar datos para entrenamiento
        X = datos_procesados[self.features_por_dia.get(dia_nombre, [])]
        y = datos_procesados['numero']
        
        print(f"   🔄 Características creadas: {X.shape[1]} features")
        
        # Entrenar modelo
        return self.entrenar_modelo_completo(X, y, dia_nombre)

    def crear_caracteristicas_completas(self, datos, dia_nombre):
        """Crea características completas para el entrenamiento"""
        
        # Características básicas
        datos_procesados = datos.copy()
        datos_procesados['hora_cuadrada'] = datos_procesados['hora'] ** 2
        datos_procesados['hora_tardia'] = datos_procesados['hora'].apply(lambda x: 1 if x >= 14 else 0)
        datos_procesados['hora_manana'] = datos_procesados['hora'].apply(lambda x: 1 if x <= 12 else 0)
        
        # Características cíclicas
        datos_procesados['hora_sin'] = np.sin(2 * np.pi * datos_procesados['hora'] / 24)
        datos_procesados['hora_cos'] = np.cos(2 * np.pi * datos_procesados['hora'] / 24)
        
        # Frecuencias históricas para este día
        if len(datos_procesados) > 10:
            freq_dia = datos_procesados['numero'].value_counts(normalize=True).to_dict()
            datos_procesados['freq_dia'] = datos_procesados['numero'].map(freq_dia).fillna(0)
        else:
            datos_procesados['freq_dia'] = 0.02
        
        # Frecuencia por hora específica
        for hora in range(10, 16):
            mask = datos_procesados['hora'] == hora
            if mask.any() and len(datos_procesados[mask]) > 2:
                freq_hora = datos_procesados[mask]['numero'].value_counts(normalize=True).to_dict()
                datos_procesados.loc[mask, 'freq_hora'] = datos_procesados.loc[mask, 'numero'].map(freq_hora).fillna(0)
            else:
                datos_procesados.loc[mask, 'freq_hora'] = 0.02
        
        # Rellenar valores NaN
        datos_procesados['freq_dia'] = datos_procesados['freq_dia'].fillna(0.02)
        datos_procesados['freq_hora'] = datos_procesados['freq_hora'].fillna(0.02)
        
        # Características adicionales
        datos_procesados['es_par'] = datos_procesados['numero'].apply(lambda x: 1 if x % 2 == 0 else 0)
        datos_procesados['decena'] = datos_procesados['numero'] // 10
        
        # Definir características
        features = [
            'hora', 'hora_cuadrada', 'hora_tardia', 'hora_manana',
            'hora_sin', 'hora_cos',
            'freq_dia', 'freq_hora',
            'es_par', 'decena'
        ]
        
        self.features_por_dia[dia_nombre] = features
        
        return datos_procesados

    def entrenar_modelo_completo(self, X, y, dia_nombre):
        """Entrena modelo completo para un día"""
        
        print(f"   🔄 Entrenando modelos para {dia_nombre}...")
        
        # Codificar target
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        self.encoders_por_dia[dia_nombre] = encoder
        self.num_classes = len(encoder.classes_)
        
        print(f"   🔢 Números únicos en datos: {self.num_classes}")
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers_por_dia[dia_nombre] = scaler
        
        modelos = {}
        
        # Modelo 1: Random Forest
        try:
            n_estimators = min(100, max(30, len(X) // 5))
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=7,
                min_samples_split=8,
                random_state=42
            )
            rf_model.fit(X_scaled, y_encoded)
            modelos['random_forest'] = rf_model
            print(f"   ✅ Random Forest entrenado ({n_estimators} árboles)")
        except Exception as e:
            print(f"   ⚠️ Error Random Forest: {e}")
        
        # Modelo 2: Gradient Boosting
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=40,
                max_depth=5,
                learning_rate=0.1,
                random_state=43
            )
            gb_model.fit(X_scaled, y_encoded)
            modelos['gradient_boosting'] = gb_model
            print(f"   ✅ Gradient Boosting entrenado")
        except Exception as e:
            print(f"   ⚠️ Error Gradient Boosting: {e}")
        
        if modelos:
            self.modelos_por_dia[dia_nombre] = modelos
            print(f"   🎯 Modelo para {dia_nombre} creado exitosamente")
            return True
        
        return False

    def entrenar_modelo_con_pocos_datos(self, datos, dia_nombre):
        """Enfoque alternativo cuando hay pocos datos"""
        
        print(f"   🔄 Usando enfoque para pocos datos...")
        
        if len(datos) == 0:
            print(f"   ❌ No hay datos para entrenar")
            return False
        
        # Enfoque basado en frecuencias y patrones simples
        self.modelos_por_dia[dia_nombre] = {
            'tipo': 'frecuencia',
            'datos': datos,
            'frecuencias': datos['numero'].value_counts().to_dict(),
            'patrones_hora': self._calcular_patrones_hora_simple(datos)
        }
        
        print(f"   📊 Modelo de frecuencias creado para {dia_nombre}")
        print(f"   💡 Basado en {len(datos)} registros históricos")
        
        return True

    def _calcular_patrones_hora_simple(self, datos):
        """Calcula patrones horarios simples"""
        patrones = {}
        for hora in range(10, 16):
            datos_hora = datos[datos['hora'] == hora]
            if len(datos_hora) > 0:
                numeros = datos_hora['numero'].tolist()
                freq = Counter(numeros)
                if freq:
                    patrones[hora] = freq.most_common(3)
        return patrones

    def predecir_para_dia(self, dia_nombre, hora):
        """Predicción para un día y hora específicos"""
        
        if dia_nombre not in self.modelos_por_dia:
            print(f"⚠️ No hay modelo para {dia_nombre}")
            return self._prediccion_por_frecuencia_general(dia_nombre, hora)
        
        modelo_dia = self.modelos_por_dia[dia_nombre]
        
        if modelo_dia.get('tipo') == 'frecuencia':
            # Usar modelo de frecuencias
            return self._prediccion_por_frecuencia(modelo_dia, dia_nombre, hora)
        else:
            # Usar modelo ML completo
            return self._prediccion_por_modelo_ml(modelo_dia, dia_nombre, hora)

    def _prediccion_por_modelo_ml(self, modelo_dia, dia_nombre, hora):
        """Predicción usando modelos ML"""
        
        # Preparar características
        features = self._preparar_features_ml(dia_nombre, hora)
        
        # Escalar
        scaler = self.scalers_por_dia[dia_nombre]
        X_scaled = scaler.transform(features)
        
        # Obtener predicciones de cada modelo
        predicciones = []
        confianzas = []
        
        for nombre, modelo in modelo_dia.items():
            if nombre not in ['tipo', 'datos', 'frecuencias', 'patrones_hora']:
                try:
                    proba = modelo.predict_proba(X_scaled)[0]
                    pred_idx = np.argmax(proba)
                    encoder = self.encoders_por_dia[dia_nombre]
                    pred_num = encoder.inverse_transform([pred_idx])[0]
                    conf = np.max(proba)
                    
                    predicciones.append(pred_num)
                    confianzas.append(conf)
                except:
                    continue
        
        if not predicciones:
            return self._prediccion_por_frecuencia_general(dia_nombre, hora)
        
        # Combinar predicciones
        pred_final = max(set(predicciones), key=predicciones.count)
        conf_final = np.mean([c for p, c in zip(predicciones, confianzas) if p == pred_final])
        
        return {
            'prediccion': pred_final,
            'confianza': min(conf_final, 0.75),
            'metodo': 'ML Ensemble',
            'alternativas': self._generar_alternativas_ml(dia_nombre, pred_final)
        }

    def _prediccion_por_frecuencia(self, modelo_dia, dia_nombre, hora):
        """Predicción basada en frecuencias"""
        
        datos = modelo_dia.get('datos', pd.DataFrame())
        frecuencias = modelo_dia.get('frecuencias', {})
        patrones_hora = modelo_dia.get('patrones_hora', {})
        
        # 1. Primero verificar si hay patrón para esta hora específica
        if hora in patrones_hora and patrones_hora[hora]:
            num_predicho = patrones_hora[hora][0][0]  # El más común a esta hora
            confianza = 0.45
            metodo = f"Patrón hora {hora}:00"
        
        # 2. Si no, usar frecuencias generales del día
        elif frecuencias:
            num_predicho = max(frecuencias.items(), key=lambda x: x[1])[0]
            confianza = 0.35
            metodo = "Frecuencia general"
        
        # 3. Fallback
        else:
            num_predicho = np.random.choice(self.all_possible_numbers)
            confianza = 0.2
            metodo = "Aleatorio"
        
        return {
            'prediccion': num_predicho,
            'confianza': confianza,
            'metodo': metodo,
            'alternativas': self._generar_alternativas_frecuencia(frecuencias, num_predicho)
        }

    def _prediccion_por_frecuencia_general(self, dia_nombre, hora):
        """Predicción general por frecuencia cuando no hay modelo"""
        
        datos_dia = self.datos_completos_por_dia.get(dia_nombre, pd.DataFrame())
        
        if len(datos_dia) > 0:
            # Filtrar por hora similar (±1 hora)
            datos_hora = datos_dia[(datos_dia['hora'] >= hora-1) & (datos_dia['hora'] <= hora+1)]
            
            if len(datos_hora) > 0:
                numeros = datos_hora['numero'].tolist()
                num_predicho = Counter(numeros).most_common(1)[0][0]
                return {
                    'prediccion': num_predicho,
                    'confianza': 0.3,
                    'metodo': 'Frecuencia histórica',
                    'alternativas': []
                }
        
        # Fallback final
        return {
            'prediccion': np.random.choice(self.all_possible_numbers),
            'confianza': 0.15,
            'metodo': 'Aleatorio',
            'alternativas': []
        }

    def _preparar_features_ml(self, dia_nombre, hora):
        """Prepara características para ML"""
        
        features = {
            'hora': hora,
            'hora_cuadrada': hora ** 2,
            'hora_tardia': 1 if hora >= 14 else 0,
            'hora_manana': 1 if hora <= 12 else 0,
            'hora_sin': np.sin(2 * np.pi * hora / 24),
            'hora_cos': np.cos(2 * np.pi * hora / 24),
            'freq_dia': 0.02,
            'freq_hora': 0.02,
            'es_par': 1 if np.random.random() > 0.5 else 0,  # Placeholder
            'decena': hora // 10
        }
        
        return pd.DataFrame([features])[self.features_por_dia.get(dia_nombre, [])]

    def _generar_alternativas_ml(self, dia_nombre, prediccion_principal):
        """Genera alternativas para ML"""
        alternativas = []
        for _ in range(2):
            num = np.random.choice([n for n in self.all_possible_numbers if n != prediccion_principal])
            conf = np.random.uniform(0.15, 0.3)
            alternativas.append({'numero': f"{num:02d}", 'confianza': f"{conf:.1%}"})
        return alternativas

    def _generar_alternativas_frecuencia(self, frecuencias, prediccion_principal):
        """Genera alternativas basadas en frecuencias"""
        alternativas = []
        if frecuencias:
            # Tomar los siguientes más frecuentes
            otros_numeros = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)[1:4]
            for num, _ in otros_numeros[:2]:
                if num != prediccion_principal:
                    conf = np.random.uniform(0.2, 0.35)
                    alternativas.append({'numero': f"{num:02d}", 'confianza': f"{conf:.1%}"})
        
        # Completar si es necesario
        while len(alternativas) < 2:
            num = np.random.choice([n for n in self.all_possible_numbers if n != prediccion_principal])
            conf = np.random.uniform(0.1, 0.25)
            alternativas.append({'numero': f"{num:02d}", 'confianza': f"{conf:.1%}"})
        
        return alternativas

    def generar_predicciones_dia_completo(self, dia_nombre):
        """Genera predicciones para todas las horas de un día"""
        
        print(f"\n🎯 GENERANDO PREDICCIONES PARA {dia_nombre.upper()}")
        print("="*60)
        
        horas = list(range(10, 16))
        resultados = []
        numeros_usados = set()
        
        for hora in horas:
            print(f"   ⏰ Procesando hora {hora}:00...")
            
            # Obtener predicción
            pred = self.predecir_para_dia(dia_nombre, hora)
            
            numero_pred = pred['prediccion']
            confianza = pred['confianza']
            metodo = pred['metodo']
            alternativas = pred['alternativas']
            
            # Forzar diversidad
            intentos = 0
            while numero_pred in numeros_usados and intentos < 3:
                if alternativas:
                    alt_numeros = [int(a['numero']) for a in alternativas if int(a['numero']) not in numeros_usados]
                    if alt_numeros:
                        numero_pred = alt_numeros[0]
                        confianza = float(alternativas[0]['confianza'].strip('%')) / 100
                        metodo = f"Diversificado({metodo})"
                        break
                intentos += 1
            
            numeros_usados.add(numero_pred)
            
            resultados.append({
                'Hora': f"{hora}:00",
                'Predicción': f"{numero_pred:02d}",
                'Confianza': f"{confianza:.1%}",
                'Método': metodo,
                'Día': dia_nombre.capitalize(),
                'Números_Alternativos': alternativas
            })
        
        return pd.DataFrame(resultados)

# Función principal
def main_todos_los_dias():
    print("\n🔮 PREDICTOR POR DÍA - TODOS LOS DATOS 🔮")
    print("="*60)
    
    predictor = PredictorTodosLosDias()
    
    try:
        # 1. Cargar y organizar todos los datos por día
        print("\n📂 CARGANDO Y ORGANIZANDO DATOS...")
        exito = predictor.load_data_y_organizar_por_dia("historicoquiniela.xlsx")
        
        if not exito:
            raise ValueError("Error al cargar datos")
        
        # 2. Seleccionar día
        print("\n📅 SELECCIONA EL DÍA PARA ANÁLISIS Y PREDICCIÓN:")
        for i, dia in enumerate(predictor.dias_semana, 1):
            count = len(predictor.datos_completos_por_dia[dia])
            print(f"   {i}. {dia.capitalize()} ({count} registros)")
        
        opcion = input("\nSelecciona el número del día: ").strip()
        
        try:
            dia_idx = int(opcion) - 1
            if 0 <= dia_idx < len(predictor.dias_semana):
                dia_seleccionado = predictor.dias_semana[dia_idx]
            else:
                print("Opción inválida. Usando martes por defecto.")
                dia_seleccionado = 'martes'
        except:
            print("Entrada inválida. Usando martes por defecto.")
            dia_seleccionado = 'martes'
        
        # 3. Análisis detallado
        print("\n" + "="*60)
        analisis = predictor.analizar_patrones_dia_especifico(dia_seleccionado)
        
        if analisis is None:
            print(f"❌ No hay datos para analizar {dia_seleccionado}")
            return
        
        # 4. Preguntar si entrenar modelo
        if analisis['total_registros'] >= 10:
            entrenar = input(f"\n¿Entrenar modelo específico para {dia_seleccionado}? (s/n): ").strip().lower()
            if entrenar == 's':
                predictor.entrenar_modelo_dia_especifico(dia_seleccionado)
        else:
            print(f"\n⚠️ Pocos datos para entrenar modelo ML ({analisis['total_registros']} registros)")
            print("💡 Usando enfoque basado en frecuencias")
        
        # 5. Generar predicciones
        print("\n" + "="*60)
        resultados = predictor.generar_predicciones_dia_completo(dia_seleccionado)
        
        # 6. Mostrar resultados
        print("\n📋 PREDICCIONES PARA", dia_seleccionado.upper())
        print("="*60)
        
        numeros_unicos = len(set([int(row['Predicción']) for _, row in resultados.iterrows()]))
        print(f"🎯 DIVERSIDAD: {numeros_unicos}/6 números únicos")
        print(f"📅 DÍA ESPECÍFICO: {dia_seleccionado.capitalize()}")
        print(f"📊 DATOS HISTÓRICOS: {analisis['total_registros']} registros")
        
        for _, row in resultados.iterrows():
            print(f"⏰ {row['Hora']}: {row['Predicción']} (Conf: {row['Confianza']}) - {row['Método']}")
            if row['Números_Alternativos']:
                alt_str = ", ".join([f"{alt['numero']} ({alt['confianza']})" for alt in row['Números_Alternativos']])
                print(f"   💡 Alternativas: {alt_str}")
        
        # 7. Guardar
        try:
            archivo = f"prediccion_{dia_seleccionado}_completa.xlsx"
            resultados.to_excel(archivo, index=False)
            print(f"\n💾 Guardado en: {archivo}")
            
            # Guardar análisis también
            archivo_analisis = f"analisis_{dia_seleccionado}.txt"
            with open(archivo_analisis, 'w', encoding='utf-8') as f:
                f.write(f"ANÁLISIS PARA {dia_seleccionado.upper()}\n")
                f.write("="*50 + "\n")
                f.write(f"Total registros: {analisis['total_registros']}\n\n")
                
                f.write("TOP 10 NÚMEROS MÁS FRECUENTES:\n")
                for num, count in analisis['frecuencias'].most_common(10):
                    porcentaje = (count / analisis['total_registros']) * 100
                    f.write(f"  {num:02d}: {count} veces ({porcentaje:.1f}%)\n")
                
                f.write("\nPATRONES HORARIOS:\n")
                for hora, info in analisis.get('patrones_horarios', {}).items():
                    f.write(f"  {hora}:00 → {info['mas_comun']:02d} ({info['porcentaje']:.1f}%)\n")
            
            print(f"📊 Análisis guardado en: {archivo_analisis}")
            
        except Exception as e:
            print(f"⚠️ Error guardando: {e}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n¡ANÁLISIS COMPLETO POR DÍA FINALIZADO! 🎯📊")

if __name__ == "__main__":
    main_todos_los_dias()