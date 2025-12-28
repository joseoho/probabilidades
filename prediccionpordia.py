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
import seaborn as sns # type: ignore
import shutil

warnings.filterwarnings('ignore')

class PredictorPatronesDiarios:
    def __init__(self, model_dir="modelos_patrones_diarios"):
        self.model = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.num_classes = 0
        self.model_trained = False
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.columnas_dias = []
        self.ultimo_dia = None
        self.all_possible_numbers = list(range(0, 37))
        
        # Patrones por día
        self.patrones_por_dia = {}
        self.datos_completos_por_dia = {}
        
        # Modelos por día
        self.modelos_por_dia = {}
        self.encoders_por_dia = {}
        self.scalers_por_dia = {}
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        print("🔮 PREDICTOR DE PATRONES DIARIOS INICIALIZADO")

    def cargar_y_organizar_datos(self, filepath):
        """Carga datos y organiza por día"""
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
                raise ValueError("Formato no soportado")
            
            print(f"✅ Archivo cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
            
            if df.shape[1] < 8:
                raise ValueError("Se necesitan al menos 8 columnas (hora + 7 días)")
            
            # Asignar nombres
            df.columns = ['hora'] + [f'dia_{i}' for i in range(1, df.shape[1])]
            self.columnas_dias = df.columns[1:]
            
            print(f"📊 Columnas identificadas: {list(df.columns)}")
            
            # Procesar hora
            df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
            df = df.dropna(subset=['hora'])
            df['hora'] = df['hora'].astype(int)
            
            # Inicializar diccionarios
            for dia in self.dias_semana:
                self.datos_completos_por_dia[dia] = []
                self.patrones_por_dia[dia] = {}
            
            # Organizar datos por día
            total_registros = 0
            
            for i, col in enumerate(self.columnas_dias):
                dia_nombre = self.dias_semana[i % 7]
                
                datos_col = df[['hora', col]].copy()
                datos_col = datos_col.dropna(subset=[col])
                
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
            
            # Calcular patrones para cada día
            self._calcular_patrones_diarios()
            
            print(f"\n📊 RESUMEN DE DATOS POR DÍA:")
            print("="*40)
            for dia in self.dias_semana:
                count = len(self.datos_completos_por_dia[dia])
                patrones = len(self.patrones_por_dia[dia].get('numeros_comunes', []))
                print(f"   {dia.capitalize():<12}: {count:>4} registros, {patrones:>2} patrones identificados")
            
            print(f"\n✅ Total registros organizados: {total_registros}")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _calcular_patrones_diarios(self):
        """Calcula patrones específicos para cada día"""
        for dia in self.dias_semana:
            datos = self.datos_completos_por_dia[dia]
            
            if len(datos) == 0:
                self.patrones_por_dia[dia] = {
                    'numeros_comunes': [],
                    'patrones_hora': {},
                    'frecuencia_general': {},
                    'estadisticas': {
                        'total': 0,
                        'promedio': 0,
                        'moda': None,
                        'rango': (0, 0)
                    }
                }
                continue
            
            numeros = datos['numero'].tolist()
            
            # 1. Números más comunes del día
            freq = Counter(numeros)
            top_numeros = freq.most_common(10)
            
            # 2. Patrones por hora
            patrones_hora = {}
            for hora in range(10, 16):
                datos_hora = datos[datos['hora'] == hora]
                if len(datos_hora) > 0:
                    numeros_hora = datos_hora['numero'].tolist()
                    if numeros_hora:
                        freq_hora = Counter(numeros_hora)
                        top_hora = freq_hora.most_common(3)
                        if top_hora:
                            patrones_hora[hora] = top_hora
            
            # 3. Frecuencia de números pares/impares
            pares = sum(1 for n in numeros if n % 2 == 0)
            impares = len(numeros) - pares
            
            # 4. Números por decenas
            decenas = {}
            for num in numeros:
                decena = num // 10
                decenas[decena] = decenas.get(decena, 0) + 1
            
            # 5. Secuencias comunes
            secuencias = self._encontrar_secuencias(datos)
            
            # 6. Horas más productivas
            horas_productivas = []
            for hora in range(10, 16):
                count_hora = len(datos[datos['hora'] == hora])
                if count_hora > 0:
                    horas_productivas.append((hora, count_hora))
            horas_productivas.sort(key=lambda x: x[1], reverse=True)
            
            # Almacenar todos los patrones
            self.patrones_por_dia[dia] = {
                'numeros_comunes': top_numeros,
                'patrones_hora': patrones_hora,
                'frecuencia_general': dict(freq.most_common(20)),
                'estadisticas': {
                    'total': len(numeros),
                    'promedio': np.mean(numeros) if numeros else 0,
                    'moda': max(set(numeros), key=numeros.count) if numeros else None,
                    'rango': (min(numeros), max(numeros)) if numeros else (0, 0),
                    'pares_vs_impares': {'pares': pares, 'impares': impares, 'ratio': pares/len(numeros) if numeros else 0},
                    'decenas': decenas
                },
                'secuencias': secuencias,
                'horas_productivas': horas_productivas[:3] if horas_productivas else [],
                'repetidos_consecutivos': self._buscar_repetidos_consecutivos(datos),
                'tendencias_temporales': self._analizar_tendencias(datos)
            }

    def _encontrar_secuencias(self, datos):
        """Encuentra secuencias de números"""
        secuencias = {}
        
        if len(datos) < 3:
            return secuencias
        
        # Ordenar por hora y número
        datos_ordenados = datos.sort_values(['hora', 'numero'])
        numeros = datos_ordenados['numero'].tolist()
        
        # Buscar secuencias aritméticas
        for i in range(len(numeros) - 2):
            diff1 = numeros[i+1] - numeros[i]
            diff2 = numeros[i+2] - numeros[i+1]
            
            if diff1 == diff2 and abs(diff1) <= 5:
                secuencia = (numeros[i], numeros[i+1], numeros[i+2])
                if secuencia not in secuencias:
                    secuencias[secuencia] = 0
                secuencias[secuencia] += 1
        
        # Buscar números que aparecen en la misma hora en días diferentes
        horas_comunes = {}
        for hora in range(10, 16):
            nums_hora = datos[datos['hora'] == hora]['numero'].tolist()
            if len(nums_hora) >= 2:
                for num in set(nums_hora):
                    if nums_hora.count(num) >= 2:
                        if hora not in horas_comunes:
                            horas_comunes[hora] = []
                        horas_comunes[hora].append(num)
        
        return {
            'aritmeticas': dict(sorted(secuencias.items(), key=lambda x: x[1], reverse=True)[:5]),
            'misma_hora': horas_comunes
        }

    def _buscar_repetidos_consecutivos(self, datos):
        """Busca números que se repiten consecutivamente"""
        repetidos = {}
        
        if len(datos) < 2:
            return repetidos
        
        datos_ordenados = datos.sort_values('hora')
        numeros = datos_ordenados['numero'].tolist()
        horas = datos_ordenados['hora'].tolist()
        
        for i in range(len(numeros) - 1):
            if numeros[i] == numeros[i+1]:
                hora1, hora2 = horas[i], horas[i+1]
                if numeros[i] not in repetidos:
                    repetidos[numeros[i]] = []
                repetidos[numeros[i]].append((hora1, hora2))
        
        return repetidos

    def _analizar_tendencias(self, datos):
        """Analiza tendencias temporales"""
        if len(datos) < 10:
            return {}
        
        datos_ordenados = datos.sort_values('hora')
        tendencias = {
            'crecimiento': [],
            'decrecimiento': [],
            'estabilidad': []
        }
        
        # Agrupar por bloques temporales
        bloques = {'mañana': [], 'tarde': []}
        for _, row in datos_ordenados.iterrows():
            if row['hora'] <= 12:
                bloques['mañana'].append(row['numero'])
            else:
                bloques['tarde'].append(row['numero'])
        
        # Comparar bloques
        if bloques['mañana'] and bloques['tarde']:
            prom_manana = np.mean(bloques['mañana'])
            prom_tarde = np.mean(bloques['tarde'])
            
            if prom_tarde > prom_manana + 5:
                tendencias['crecimiento'].append(f"Los números aumentan en la tarde (+{prom_tarde-prom_manana:.1f})")
            elif prom_tarde < prom_manana - 5:
                tendencias['decrecimiento'].append(f"Los números disminuyen en la tarde ({prom_manana-prom_tarde:.1f})")
            else:
                tendencias['estabilidad'].append("Los números se mantienen estables")
        
        return tendencias

    def mostrar_patrones_dia(self, dia_nombre):
        """Muestra los patrones detectados para un día específico"""
        
        print(f"\n🔍 PATRONES DETECTADOS PARA {dia_nombre.upper()}")
        print("="*60)
        
        patrones = self.patrones_por_dia[dia_nombre]
        estadisticas = patrones['estadisticas']
        
        if estadisticas['total'] == 0:
            print("❌ No hay datos disponibles para este día")
            return
        
        print(f"📊 ESTADÍSTICAS GENERALES:")
        print(f"   Total registros: {estadisticas['total']}")
        print(f"   Promedio numérico: {estadisticas['promedio']:.1f}")
        print(f"   Número más común (moda): {estadisticas['moda']}")
        print(f"   Rango: {estadisticas['rango'][0]} - {estadisticas['rango'][1]}")
        
        ratio_pares = estadisticas['pares_vs_impares']['ratio']
        print(f"   Distribución pares/impares: {ratio_pares:.1%} pares")
        
        print(f"\n🎯 TOP 10 NÚMEROS MÁS FRECUENTES:")
        for i, (num, count) in enumerate(patrones['numeros_comunes'], 1):
            porcentaje = (count / estadisticas['total']) * 100
            estrellas = "⭐" * (min(5, int(porcentaje / 10)))
            print(f"   {i:2}. {num:02d}: {count:3d} veces ({porcentaje:5.1f}%) {estrellas}")
        
        print(f"\n⏰ PATRONES POR HORA:")
        for hora, top_nums in patrones['patrones_hora'].items():
            if top_nums:
                nums_str = ", ".join([f"{num}({count})" for num, count in top_nums])
                print(f"   {hora}:00 → {nums_str}")
        
        print(f"\n📈 TENDENCIAS IDENTIFICADAS:")
        if patrones['secuencias']['aritmeticas']:
            print(f"   Secuencias aritméticas:")
            for secuencia, count in patrones['secuencias']['aritmeticas'].items():
                print(f"     {secuencia} (aparece {count} veces)")
        
        if patrones['secuencias']['misma_hora']:
            print(f"\n   Números que se repiten en la misma hora:")
            for hora, numeros in patrones['secuencias']['misma_hora'].items():
                print(f"     Hora {hora}:00 → {', '.join(map(str, numeros))}")
        
        if patrones['repetidos_consecutivos']:
            print(f"\n   Repeticiones consecutivas:")
            for num, horas_list in patrones['repetidos_consecutivos'].items():
                print(f"     Número {num} se repitió en: {', '.join([f'{h1}-{h2}' for h1, h2 in horas_list])}")
        
        if patrones['horas_productivas']:
            print(f"\n   HORAS MÁS PRODUCTIVAS (con más registros):")
            for hora, count in patrones['horas_productivas']:
                print(f"     {hora}:00 → {count} registros")
        
        if patrones['tendencias_temporales']:
            print(f"\n   TENDENCIAS TEMPORALES:")
            for categoria, mensajes in patrones['tendencias_temporales'].items():
                for msg in mensajes:
                    print(f"     • {msg}")
        
        # Mostrar distribución por decenas
        print(f"\n🔢 DISTRIBUCIÓN POR DECENAS:")
        for decena, count in sorted(estadisticas['decenas'].items()):
            porcentaje = (count / estadisticas['total']) * 100
            barra = "█" * int(porcentaje / 5)
            print(f"   {decena}0-{decena}9: {count:3d} ({porcentaje:5.1f}%) {barra}")

    def generar_predicciones_basadas_en_patrones(self, dia_nombre):
        """Genera predicciones basadas en los patrones del día"""
        
        print(f"\n🎯 PREDICCIONES BASADAS EN PATRONES PARA {dia_nombre.upper()}")
        print("="*60)
        
        patrones = self.patrones_por_dia[dia_nombre]
        
        if patrones['estadisticas']['total'] == 0:
            print("❌ No hay patrones suficientes para generar predicciones")
            return None
        
        horas_prediccion = list(range(10, 16))
        resultados = []
        
        for hora in horas_prediccion:
            prediccion = self._predecir_hora_por_patrones(dia_nombre, hora)
            resultados.append({
                'Hora': f"{hora}:00",
                'Predicción_Principal': prediccion['principal']['numero'],
                'Confianza': prediccion['principal']['confianza'],
                'Método': prediccion['principal']['metodo'],
                'Alternativa_1': prediccion['alternativas'][0]['numero'],
                'Conf_Alt1': prediccion['alternativas'][0]['confianza'],
                'Alternativa_2': prediccion['alternativas'][1]['numero'],
                'Conf_Alt2': prediccion['alternativas'][1]['confianza'],
                'Patrón_Usado': prediccion['principal']['patron_explicacion']
            })
            
            print(f"⏰ {hora}:00 → {prediccion['principal']['numero']} "
                  f"(Conf: {prediccion['principal']['confianza']:.0%}) "
                  f"- {prediccion['principal']['metodo']}")
            print(f"   💡 Alternativas: {prediccion['alternativas'][0]['numero']} "
                  f"({prediccion['alternativas'][0]['confianza']:.0%}), "
                  f"{prediccion['alternativas'][1]['numero']} "
                  f"({prediccion['alternativas'][1]['confianza']:.0%})")
        
        df_resultados = pd.DataFrame(resultados)
        
        # Guardar resultados
        fecha_hora = datetime.now().strftime("%Y%m%d_%H%M")
        archivo_prediccion = f"prediccion_patrones_{dia_nombre}_{fecha_hora}.xlsx"
        df_resultados.to_excel(archivo_prediccion, index=False)
        print(f"\n💾 Predicciones guardadas en: {archivo_prediccion}")
        
        # Guardar reporte de patrones
        self._guardar_reporte_patrones(dia_nombre)
        
        return df_resultados

    def _predecir_hora_por_patrones(self, dia_nombre, hora):
        """Predice para una hora específica usando patrones"""
        patrones = self.patrones_por_dia[dia_nombre]
        
        # Estrategia 1: Patrón específico de la hora
        if hora in patrones['patrones_hora'] and patrones['patrones_hora'][hora]:
            num_comun = patrones['patrones_hora'][hora][0][0]
            return {
                'principal': {
                    'numero': f"{num_comun:02d}",
                    'confianza': 0.45,
                    'metodo': 'Patrón horario',
                    'patron_explicacion': f"El {num_comun:02d} aparece {patrones['patrones_hora'][hora][0][1]} veces a esta hora"
                },
                'alternativas': self._generar_alternativas(patrones, num_comun)
            }
        
        # Estrategia 2: Números más comunes del día
        if patrones['numeros_comunes']:
            num_comun = patrones['numeros_comunes'][0][0]
            return {
                'principal': {
                    'numero': f"{num_comun:02d}",
                    'confianza': 0.35,
                    'metodo': 'Número más frecuente del día',
                    'patron_explicacion': f"El {num_comun:02d} es el más común ({patrones['numeros_comunes'][0][1]} apariciones)"
                },
                'alternativas': self._generar_alternativas(patrones, num_comun)
            }
        
        # Estrategia 3: Moda del día
        if patrones['estadisticas']['moda'] is not None:
            moda = patrones['estadisticas']['moda']
            return {
                'principal': {
                    'numero': f"{moda:02d}",
                    'confianza': 0.3,
                    'metodo': 'Moda estadística',
                    'patron_explicacion': f"El {moda:02d} es el valor modal del día"
                },
                'alternativas': self._generar_alternativas(patrones, moda)
            }
        
        # Fallback: aleatorio
        num_random = np.random.choice(self.all_possible_numbers)
        return {
            'principal': {
                'numero': f"{num_random:02d}",
                'confianza': 0.15,
                'metodo': 'Aleatorio (sin patrones)',
                'patron_explicacion': 'Sin patrones detectados'
            },
            'alternativas': [
                {'numero': f"{(num_random + 1) % 37:02d}", 'confianza': 0.12},
                {'numero': f"{(num_random - 1) % 37:02d}", 'confianza': 0.10}
            ]
        }

    def _generar_alternativas(self, patrones, num_principal):
        """Genera números alternativos"""
        alternativas = []
        
        # Usar otros números comunes del día
        otros_comunes = [num for num, _ in patrones['numeros_comunes'] 
                        if num != num_principal][:3]
        
        for i, num in enumerate(otros_comunes[:2]):
            alternativas.append({
                'numero': f"{num:02d}",
                'confianza': 0.25 - (i * 0.05)
            })
        
        # Completar si es necesario
        while len(alternativas) < 2:
            num = np.random.choice([n for n in self.all_possible_numbers 
                                   if n != num_principal and n not in [int(a['numero']) for a in alternativas]])
            alternativas.append({
                'numero': f"{num:02d}",
                'confianza': 0.15
            })
        
        return alternativas

    def _guardar_reporte_patrones(self, dia_nombre):
        """Guarda un reporte detallado de los patrones"""
        patrones = self.patrones_por_dia[dia_nombre]
        
        archivo_reporte = f"reporte_patrones_{dia_nombre}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        
        with open(archivo_reporte, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DE PATRONES - {dia_nombre.upper()}\n")
            f.write("="*60 + "\n\n")
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write("-"*40 + "\n")
            for key, value in patrones['estadisticas'].items():
                if key != 'decenas' and key != 'pares_vs_impares':
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nTOP 10 NÚMEROS MÁS FRECUENTES:\n")
            f.write("-"*40 + "\n")
            for num, count in patrones['numeros_comunes']:
                porcentaje = (count / patrones['estadisticas']['total']) * 100
                f.write(f"  {num:02d}: {count:4d} veces ({porcentaje:.1f}%)\n")
            
            f.write("\nPATRONES POR HORA:\n")
            f.write("-"*40 + "\n")
            for hora, top_nums in patrones['patrones_hora'].items():
                if top_nums:
                    nums_str = ", ".join([f"{num}({count})" for num, count in top_nums])
                    f.write(f"  {hora}:00 → {nums_str}\n")
            
            if patrones['secuencias']['aritmeticas']:
                f.write("\nSECUENCIAS ARITMÉTICAS DETECTADAS:\n")
                f.write("-"*40 + "\n")
                for secuencia, count in patrones['secuencias']['aritmeticas'].items():
                    f.write(f"  {secuencia}: aparece {count} veces\n")
            
            if patrones['repetidos_consecutivos']:
                f.write("\nREPETICIONES CONSECUTIVAS:\n")
                f.write("-"*40 + "\n")
                for num, horas_list in patrones['repetidos_consecutivos'].items():
                    f.write(f"  Número {num}: {len(horas_list)} repeticiones\n")
            
            f.write("\nHORAS MÁS PRODUCTIVAS:\n")
            f.write("-"*40 + "\n")
            for hora, count in patrones['horas_productivas']:
                f.write(f"  {hora}:00 → {count} registros\n")
        
        print(f"📄 Reporte de patrones guardado en: {archivo_reporte}")

def main_patrones_diarios():
    """Función principal"""
    print("\n" + "="*70)
    print("🔮 SISTEMA DE DETECCIÓN DE PATRONES DIARIOS")
    print("="*70)
    
    predictor = PredictorPatronesDiarios()
    
    try:
        # Cargar datos
        print("\n📥 CARGANDO DATOS HISTÓRICOS...")
        archivo = "historicoquiniela.xlsx"
        
        if not os.path.exists(archivo):
            print(f"❌ No se encontró el archivo: {archivo}")
            print("💡 Asegúrate de que el archivo 'historicoquiniela.xlsx' esté en la misma carpeta")
            return
        
        exito = predictor.cargar_y_organizar_datos(archivo)
        
        if not exito:
            print("❌ Error al cargar los datos")
            return
        
        # Menú de selección
        while True:
            print("\n" + "="*60)
            print("📅 MENÚ DE PATRONES DIARIOS")
            print("="*60)
            print("Selecciona una opción:")
            print("1. Ver patrones de un día específico")
            print("2. Generar predicciones basadas en patrones")
            print("3. Ver resumen de todos los días")
            print("4. Salir")
            print("-"*40)
            
            opcion = input("Tu elección (1-4): ").strip()
            
            if opcion == '1':
                print("\nSelecciona el día para analizar:")
                for i, dia in enumerate(predictor.dias_semana, 1):
                    count = len(predictor.datos_completos_por_dia[dia])
                    print(f"  {i}. {dia.capitalize():<12} ({count:>4} registros)")
                
                try:
                    dia_idx = int(input("\nNúmero del día: ")) - 1
                    if 0 <= dia_idx < 7:
                        dia_seleccionado = predictor.dias_semana[dia_idx]
                        predictor.mostrar_patrones_dia(dia_seleccionado)
                    else:
                        print("❌ Opción inválida")
                except ValueError:
                    print("❌ Por favor ingresa un número válido")
            
            elif opcion == '2':
                print("\nSelecciona el día para predicción:")
                for i, dia in enumerate(predictor.dias_semana, 1):
                    count = len(predictor.datos_completos_por_dia[dia])
                    print(f"  {i}. {dia.capitalize():<12} ({count:>4} registros)")
                
                try:
                    dia_idx = int(input("\nNúmero del día: ")) - 1
                    if 0 <= dia_idx < 7:
                        dia_seleccionado = predictor.dias_semana[dia_idx]
                        predictor.generar_predicciones_basadas_en_patrones(dia_seleccionado)
                    else:
                        print("❌ Opción inválida")
                except ValueError:
                    print("❌ Por favor ingresa un número válido")
            
            elif opcion == '3':
                print("\n📊 RESUMEN DE TODOS LOS DÍAS:")
                print("="*60)
                for dia in predictor.dias_semana:
                    patrones = predictor.patrones_por_dia[dia]
                    total = patrones['estadisticas']['total']
                    if total > 0:
                        top_num = patrones['numeros_comunes'][0][0] if patrones['numeros_comunes'] else 'N/A'
                        top_count = patrones['numeros_comunes'][0][1] if patrones['numeros_comunes'] else 0
                        print(f"  {dia.capitalize():<12}: {total:>4} registros | "
                              f"Top: {top_num:02d} ({top_count} veces) | "
                              f"Moda: {patrones['estadisticas']['moda']}")
                    else:
                        print(f"  {dia.capitalize():<12}: Sin datos")
            
            elif opcion == '4':
                print("\n👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción no válida. Por favor elige 1-4.")
                
            # Preguntar si continuar
            if opcion != '4':
                continuar = input("\n¿Continuar con otra operación? (s/n): ").strip().lower()
                if continuar != 's':
                    print("\n👋 ¡Hasta luego!")
                    break
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_patrones_diarios()