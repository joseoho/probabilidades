import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SistemaPatronesAvanzado:
    def __init__(self):
        self.dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
        self.all_numbers = list(range(0, 37))
        self.datos_por_dia = {dia: pd.DataFrame() for dia in self.dias_semana}
        self.patrones_avanzados = {dia: {} for dia in self.dias_semana}
        
    def cargar_datos(self, archivo):
        """Carga y procesa los datos del archivo"""
        print(f"📂 Cargando datos de {archivo}...")
        # Tu código de carga existente
        # ...
        return True
    
    def analizar_patrones_avanzados(self, dia_nombre):
        """Análisis avanzado de patrones específicos"""
        
        print(f"\n🔍 ANÁLISIS AVANZADO PARA {dia_nombre.upper()}")
        print("="*70)
        
        datos = self.datos_por_dia[dia_nombre]
        if len(datos) == 0:
            print("No hay datos para analizar")
            return
        
        # Análisis de repeticiones mejorado
        repeticiones_analizadas = self._analizar_repeticiones_detallado(datos)
        
        # Identificar números "calientes" y "fríos"
        numeros_calientes = self._identificar_numeros_calientes(repeticiones_analizadas)
        
        # Análisis de secuencias temporales
        secuencias_temporales = self._analizar_secuencias_temporales(datos)
        
        # Patrones de hora específica
        patrones_hora_especifica = self._analizar_patrones_hora_especifica(datos)
        
        # Guardar análisis
        self.patrones_avanzados[dia_nombre] = {
            'repeticiones': repeticiones_analizadas,
            'numeros_calientes': numeros_calientes,
            'secuencias': secuencias_temporales,
            'patrones_hora': patrones_hora_especifica,
            'resumen': self._crear_resumen_avanzado(datos, repeticiones_analizadas, numeros_calientes)
        }
        
        # Mostrar resultados
        self._mostrar_resultados_avanzados(dia_nombre)
        
    def _analizar_repeticiones_detallado(self, datos):
        """Análisis detallado de repeticiones"""
        
        print("📊 ANALIZANDO REPETICIONES...")
        
        repeticiones = {
            'misma_hora_exacta': defaultdict(list),  # Mismo número en misma hora
            'horas_consecutivas': defaultdict(list),  # Mismo número en horas consecutivas
            'mismo_numero_diferentes_horas': defaultdict(list),  # Mismo número en horas diferentes
            'patrones_horarios': defaultdict(lambda: defaultdict(int))  # Qué números a qué horas
        }
        
        # Ordenar por hora
        datos_ordenados = datos.sort_values('hora')
        horas = datos_ordenados['hora'].tolist()
        numeros = datos_ordenados['numero'].tolist()
        
        # Análisis 1: Repeticiones en la misma hora
        for i in range(len(numeros) - 1):
            for j in range(i + 1, len(numeros)):
                if horas[i] == horas[j] and numeros[i] == numeros[j]:
                    num = numeros[i]
                    hora = horas[i]
                    repeticiones['misma_hora_exacta'][num].append(hora)
                    
                    # Registrar en patrones horarios
                    repeticiones['patrones_horarios'][hora][num] += 1
        
        # Análisis 2: Números en horas consecutivas
        for i in range(len(numeros) - 1):
            if numeros[i] == numeros[i+1]:
                num = numeros[i]
                par_horas = (horas[i], horas[i+1])
                repeticiones['horas_consecutivas'][num].append(par_horas)
        
        # Análisis 3: Mismo número en diferentes horas
        hora_por_numero = defaultdict(set)
        for hora, num in zip(horas, numeros):
            hora_por_numero[num].add(hora)
        
        for num, horas_set in hora_por_numero.items():
            if len(horas_set) > 1:
                repeticiones['mismo_numero_diferentes_horas'][num] = sorted(list(horas_set))
        
        # Calcular estadísticas
        estadisticas = {
            'total_repeticiones_misma_hora': sum(len(v) for v in repeticiones['misma_hora_exacta'].values()),
            'numeros_con_repeticion_misma_hora': len(repeticiones['misma_hora_exacta']),
            'horas_con_mas_repeticiones': self._calcular_horas_activas(repeticiones['patrones_horarios']),
            'numeros_mas_repetidos': self._calcular_numeros_mas_repetidos(repeticiones)
        }
        
        repeticiones['estadisticas'] = estadisticas
        
        return repeticiones
    
    def _calcular_horas_activas(self, patrones_horarios):
        """Calcula las horas con más actividad de repetición"""
        horas_actividad = []
        for hora, numeros_dict in patrones_horarios.items():
            total_repeticiones = sum(numeros_dict.values())
            if total_repeticiones > 0:
                horas_actividad.append((hora, total_repeticiones, numeros_dict))
        
        # Ordenar por actividad
        horas_actividad.sort(key=lambda x: x[1], reverse=True)
        return horas_actividad
    
    def _calcular_numeros_mas_repetidos(self, repeticiones):
        """Calcula los números más repetidos"""
        numeros_repeticion = []
        
        for num in set(list(repeticiones['misma_hora_exacta'].keys()) + 
                      list(repeticiones['horas_consecutivas'].keys())):
            
            rep_misma_hora = len(repeticiones['misma_hora_exacta'].get(num, []))
            rep_consecutivas = len(repeticiones['horas_consecutivas'].get(num, []))
            total = rep_misma_hora + rep_consecutivas
            
            if total > 0:
                horas_unicas = set(repeticiones['misma_hora_exacta'].get(num, []))
                horas_unicas.update([h for h1, h2 in repeticiones['horas_consecutivas'].get(num, []) 
                                   for h in (h1, h2)])
                
                numeros_repeticion.append({
                    'numero': num,
                    'repeticiones_misma_hora': rep_misma_hora,
                    'repeticiones_consecutivas': rep_consecutivas,
                    'total_repeticiones': total,
                    'horas_activas': sorted(list(horas_unicas)),
                    'horas_count': len(horas_unicas),
                    'score': total * len(horas_unicas)  # Puntuación combinada
                })
        
        # Ordenar por score
        numeros_repeticion.sort(key=lambda x: x['score'], reverse=True)
        return numeros_repeticion
    
    def _identificar_numeros_calientes(self, repeticiones):
        """Identifica números 'calientes' (con patrones fuertes)"""
        
        numeros_calientes = []
        numeros_mas_repetidos = repeticiones.get('numeros_mas_repetidos', [])
        
        for num_info in numeros_mas_repetidos:
            num = num_info['numero']
            total_rep = num_info['total_repeticiones']
            horas_count = num_info['horas_count']
            
            # Criterios para ser "caliente"
            es_caliente = False
            razones = []
            
            if total_rep >= 3:
                es_caliente = True
                razones.append(f"Se repite {total_rep} veces")
            
            if horas_count >= 2:
                es_caliente = True
                razones.append(f"Aparece en {horas_count} horas diferentes")
            
            if num_info['repeticiones_misma_hora'] >= 2:
                es_caliente = True
                razones.append(f"Se repite {num_info['repeticiones_misma_hora']} veces en la misma hora")
            
            if es_caliente:
                numeros_calientes.append({
                    'numero': num,
                    'razones': razones,
                    'score': num_info['score'],
                    'detalles': num_info
                })
        
        return numeros_calientes
    
    def _analizar_secuencias_temporales(self, datos):
        """Analiza secuencias y tendencias temporales"""
        
        secuencias = {
            'numeros_que_aumentan': [],
            'numeros_que_disminuyen': [],
            'patrones_mañana_tarde': {'mañana': [], 'tarde': []},
            'ventanas_temporales': defaultdict(list)
        }
        
        if len(datos) < 10:
            return secuencias
        
        # Agrupar por bloques de 2 horas
        for hora_inicio in range(10, 15, 2):
            hora_fin = hora_inicio + 2
            mask = (datos['hora'] >= hora_inicio) & (datos['hora'] < hora_fin)
            datos_ventana = datos[mask]
            
            if len(datos_ventana) > 0:
                numeros = datos_ventana['numero'].tolist()
                frecuencias = Counter(numeros)
                
                if frecuencias:
                    num_comun = frecuencias.most_common(1)[0][0]
                    secuencias['ventanas_temporales'][f"{hora_inicio}-{hora_fin-1}"].append({
                        'numero_comun': num_comun,
                        'frecuencia': frecuencias.most_common(1)[0][1],
                        'total_numeros': len(datos_ventana)
                    })
        
        # Analizar transición mañana-tarde
        datos_manana = datos[datos['hora'] <= 12]
        datos_tarde = datos[datos['hora'] > 12]
        
        if len(datos_manana) > 0 and len(datos_tarde) > 0:
            freq_manana = Counter(datos_manana['numero'].tolist())
            freq_tarde = Counter(datos_tarde['numero'].tolist())
            
            # Encontrar números que aumentan
            for num in set(list(freq_manana.keys()) + list(freq_tarde.keys())):
                manana_count = freq_manana.get(num, 0)
                tarde_count = freq_tarde.get(num, 0)
                
                if manana_count > 0 and tarde_count > 0:
                    if tarde_count > manana_count * 1.5:  # Aumenta 50% o más
                        secuencias['numeros_que_aumentan'].append({
                            'numero': num,
                            'manana': manana_count,
                            'tarde': tarde_count,
                            'incremento': ((tarde_count - manana_count) / manana_count) * 100
                        })
                    elif tarde_count < manana_count * 0.7:  # Disminuye 30% o más
                        secuencias['numeros_que_disminuyen'].append({
                            'numero': num,
                            'manana': manana_count,
                            'tarde': tarde_count,
                            'decremento': ((manana_count - tarde_count) / manana_count) * 100
                        })
        
        return secuencias
    
    def _analizar_patrones_hora_especifica(self, datos):
        """Analiza patrones específicos por hora"""
        
        patrones = {}
        
        for hora in range(10, 16):
            datos_hora = datos[datos['hora'] == hora]
            
            if len(datos_hora) > 0:
                numeros = datos_hora['numero'].tolist()
                total = len(numeros)
                
                if total > 0:
                    # Frecuencia
                    frecuencia = Counter(numeros)
                    top_3 = frecuencia.most_common(3)
                    
                    # Paridad
                    pares = sum(1 for n in numeros if n % 2 == 0)
                    impares = total - pares
                    
                    # Decenas
                    decenas_dist = Counter([n // 10 for n in numeros])
                    
                    patrones[hora] = {
                        'total_registros': total,
                        'top_numeros': top_3,
                        'probabilidad_top': [(num, count/total) for num, count in top_3],
                        'paridad': {'pares': pares, 'impares': impares, 'ratio_pares': pares/total},
                        'decenas': dict(decenas_dist),
                        'promedio': np.mean(numeros) if numeros else 0,
                        'rango': (min(numeros), max(numeros)) if numeros else (0, 0)
                    }
        
        return patrones
    
    def _crear_resumen_avanzado(self, datos, repeticiones, numeros_calientes):
        """Crea un resumen avanzado del análisis"""
        
        total_registros = len(datos)
        
        resumen = {
            'total_registros': total_registros,
            'horas_activas': [hora for hora, _, _ in repeticiones.get('estadisticas', {}).get('horas_con_mas_repeticiones', [])[:3]],
            'numeros_calientes_count': len(numeros_calientes),
            'numeros_calientes_top': [n['numero'] for n in numeros_calientes[:5]],
            'repeticiones_totales': repeticiones.get('estadisticas', {}).get('total_repeticiones_misma_hora', 0),
            'horas_con_mas_actividad': self._resumen_horas_activas(repeticiones),
            'recomendaciones': self._generar_recomendaciones(repeticiones, numeros_calientes)
        }
        
        return resumen
    
    def _resumen_horas_activas(self, repeticiones):
        """Crea resumen de horas más activas"""
        horas_activas = repeticiones.get('estadisticas', {}).get('horas_con_mas_repeticiones', [])
        resumen = []
        
        for hora, total, numeros_dict in horas_activas[:3]:
            top_numeros = sorted(numeros_dict.items(), key=lambda x: x[1], reverse=True)[:2]
            numeros_str = ", ".join([f"{num}({count})" for num, count in top_numeros])
            resumen.append(f"{hora}:00: {total} repeticiones ({numeros_str})")
        
        return resumen
    
    def _generar_recomendaciones(self, repeticiones, numeros_calientes):
        """Genera recomendaciones basadas en patrones"""
        recomendaciones = []
        
        # Recomendaciones basadas en números calientes
        if numeros_calientes:
            for num_info in numeros_calientes[:3]:
                num = num_info['numero']
                razones = ", ".join(num_info['razones'][:2])
                recomendaciones.append(f"✅ Considerar el {num:02d}: {razones}")
        
        # Recomendaciones basadas en horas
        horas_activas = repeticiones.get('estadisticas', {}).get('horas_con_mas_repeticiones', [])
        if horas_activas:
            hora_mas_activa = horas_activas[0][0]
            recomendaciones.append(f"⏰ Hora crítica: {hora_mas_activa}:00 tiene más repeticiones")
        
        # Recomendaciones generales
        total_rep = repeticiones.get('estadisticas', {}).get('total_repeticiones_misma_hora', 0)
        if total_rep > 10:
            recomendaciones.append(f"🎯 Alto índice de repetición: {total_rep} repeticiones en misma hora")
        
        return recomendaciones
    
    def _mostrar_resultados_avanzados(self, dia_nombre):
        """Muestra los resultados del análisis avanzado"""
        
        patrones = self.patrones_avanzados[dia_nombre]
        repeticiones = patrones['repeticiones']
        numeros_calientes = patrones['numeros_calientes']
        resumen = patrones['resumen']
        
        print(f"\n🎯 ANÁLISIS AVANZADO - {dia_nombre.upper()}")
        print("="*70)
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   • Total registros: {resumen['total_registros']}")
        print(f"   • Repeticiones en misma hora: {resumen['repeticiones_totales']}")
        print(f"   • Números 'calientes' identificados: {resumen['numeros_calientes_count']}")
        
        print(f"\n🔥 NÚMEROS 'CALIENTES' (con patrones fuertes):")
        if numeros_calientes:
            for i, num_info in enumerate(numeros_calientes[:5], 1):
                num = num_info['numero']
                score = num_info['score']
                razones = " | ".join(num_info['razones'][:2])
                print(f"   {i}. {num:02d} (Score: {score:.1f}): {razones}")
        else:
            print("   No se identificaron números con patrones fuertes")
        
        print(f"\n⏰ HORAS MÁS ACTIVAS:")
        if 'horas_con_mas_actividad' in resumen:
            for hora_info in resumen['horas_con_mas_actividad'][:3]:
                print(f"   • {hora_info}")
        
        print(f"\n📈 RECOMENDACIONES:")
        for i, rec in enumerate(resumen.get('recomendaciones', [])[:5], 1):
            print(f"   {i}. {rec}")
        
        # Mostrar análisis de repeticiones detallado
        print(f"\n🔍 DETALLE DE REPETICIONES:")
        estadisticas = repeticiones.get('estadisticas', {})
        if 'numeros_mas_repetidos' in estadisticas:
            print("   Top números más repetidos:")
            for num_info in estadisticas['numeros_mas_repetidos'][:5]:
                num = num_info['numero']
                total = num_info['total_repeticiones']
                horas = num_info['horas_activas']
                print(f"      {num:02d}: {total} repeticiones en horas {horas}")
    
    def predecir_con_analisis_avanzado(self, dia_nombre, hora):
        """Predicción usando análisis avanzado"""
        
        print(f"\n🎯 PREDICCIÓN AVANZADA - {dia_nombre.upper()} {hora}:00")
        print("="*70)
        
        if dia_nombre not in self.patrones_avanzados:
            print("   ℹ️ Primero ejecuta análisis avanzado para este día")
            return None
        
        patrones = self.patrones_avanzados[dia_nombre]
        
        # 1. Buscar números calientes para esta hora
        numeros_recomendados = []
        
        # Verificar números calientes que han aparecido en esta hora
        for num_info in patrones['numeros_calientes']:
            num = num_info['numero']
            detalles = num_info['detalles']
            
            # Si este número ha aparecido en esta hora
            if hora in detalles['horas_activas']:
                # Calcular confianza
                confianza_base = 0.4
                
                # Aumentar confianza si se repitió en esta hora
                rep_esta_hora = detalles['horas_activas'].count(hora)
                if rep_esta_hora > 0:
                    confianza_base += min(0.3, rep_esta_hora * 0.1)
                
                # Aumentar si tiene muchas repeticiones totales
                if detalles['total_repeticiones'] >= 3:
                    confianza_base += 0.1
                
                numeros_recomendados.append({
                    'numero': num,
                    'confianza': min(0.75, confianza_base),
                    'razones': num_info['razones'],
                    'tipo': 'Número caliente con historial en esta hora'
                })
        
        # 2. Verificar patrones de hora específica
        if hora in patrones['patrones_hora']:
            info_hora = patrones['patrones_hora'][hora]
            top_numeros = info_hora['top_numeros']
            
            for num, count in top_numeros:
                # Si no está ya en recomendados
                if not any(r['numero'] == num for r in numeros_recomendados):
                    probabilidad = count / info_hora['total_registros']
                    confianza = 0.3 + (probabilidad * 0.3)
                    
                    numeros_recomendados.append({
                        'numero': num,
                        'confianza': min(0.7, confianza),
                        'razones': [f"Top {top_numeros.index((num, count))+1} a esta hora ({count} apariciones)"],
                        'tipo': 'Patrón horario específico'
                    })
        
        # 3. Ordenar por confianza
        numeros_recomendados.sort(key=lambda x: x['confianza'], reverse=True)
        
        # 4. Preparar resultado
        if numeros_recomendados:
            mejor = numeros_recomendados[0]
            
            resultado = {
                'prediccion_principal': {
                    'numero': f"{mejor['numero']:02d}",
                    'confianza': mejor['confianza'],
                    'tipo': mejor['tipo'],
                    'explicacion': "; ".join(mejor['razones'][:2])
                },
                'alternativas': [],
                'analisis_usado': {
                    'total_numeros_considerados': len(numeros_recomendados),
                    'nivel_confianza': 'ALTO' if mejor['confianza'] > 0.6 else 'MEDIO',
                    'patrones_aplicados': len([n for n in numeros_recomendados if 'caliente' in n['tipo']])
                }
            }
            
            # Agregar alternativas
            for alt in numeros_recomendados[1:4]:
                resultado['alternativas'].append({
                    'numero': f"{alt['numero']:02d}",
                    'confianza': alt['confianza'],
                    'razon': alt['razones'][0] if alt['razones'] else "Patrón secundario"
                })
            
            # Mostrar resultado
            print(f"\n   🎯 PREDICCIÓN PRINCIPAL: {resultado['prediccion_principal']['numero']}")
            print(f"   📊 Confianza: {resultado['prediccion_principal']['confianza']:.0%}")
            print(f"   🔍 Tipo: {resultado['prediccion_principal']['tipo']}")
            print(f"   📝 Explicación: {resultado['prediccion_principal']['explicacion']}")
            
            if resultado['alternativas']:
                print(f"\n   💡 ALTERNATIVAS RECOMENDADAS:")
                for alt in resultado['alternativas']:
                    print(f"      • {alt['numero']} (Conf: {alt['confianza']:.0%}): {alt['razon']}")
            
            return resultado
        
        else:
            print("   ⚠️ No se encontraron patrones suficientes para esta hora")
            return None

# Función principal mejorada
def sistema_completo_avanzado():
    """Sistema completo con análisis avanzado"""
    
    print("\n" + "="*80)
    print("🎯 SISTEMA DE ANÁLISIS AVANZADO DE PATRONES DIARIOS")
    print("="*80)
    
    sistema = SistemaPatronesAvanzado()
    
    # Cargar datos
    sistema.cargar_datos("historicoquiniela.xlsx")
    
    while True:
        print("\n📋 MENÚ PRINCIPAL - ANÁLISIS AVANZADO")
        print("="*50)
        print("1. Analizar patrones de un día específico")
        print("2. Generar predicción avanzada")
        print("3. Ver resumen de análisis guardados")
        print("4. Comparar días")
        print("5. Salir")
        print("-"*50)
        
        opcion = input("Selecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            print("\nSelecciona el día a analizar:")
            for i, dia in enumerate(sistema.dias_semana, 1):
                count = len(sistema.datos_por_dia[dia])
                print(f"  {i}. {dia.capitalize()} ({count} registros)")
            
            try:
                idx = int(input("\nNúmero del día: ")) - 1
                if 0 <= idx < 7:
                    dia_seleccionado = sistema.dias_semana[idx]
                    sistema.analizar_patrones_avanzados(dia_seleccionado)
                else:
                    print("❌ Opción inválida")
            except:
                print("❌ Entrada no válida")
        
        elif opcion == '2':
            print("\nSelecciona el día para predicción:")
            for i, dia in enumerate(sistema.dias_semana, 1):
                count = len(sistema.datos_por_dia[dia])
                print(f"  {i}. {dia.capitalize()} ({count} registros)")
            
            try:
                idx = int(input("\nNúmero del día: ")) - 1
                if 0 <= idx < 7:
                    dia_seleccionado = sistema.dias_semana[idx]
                    
                    # Seleccionar hora
                    hora = int(input("Hora para predicción (10-15): "))
                    if 10 <= hora <= 15:
                        sistema.predecir_con_analisis_avanzado(dia_seleccionado, hora)
                    else:
                        print("❌ Hora debe estar entre 10 y 15")
                else:
                    print("❌ Opción inválida")
            except:
                print("❌ Entrada no válida")
        
        elif opcion == '3':
            print("\n📊 RESUMEN DE ANÁLISIS GUARDADOS:")
            print("="*50)
            for dia in sistema.dias_semana:
                if sistema.patrones_avanzados[dia]:
                    resumen = sistema.patrones_avanzados[dia].get('resumen', {})
                    if resumen:
                        print(f"  {dia.capitalize():<12}: {resumen.get('total_registros', 0)} registros | "
                              f"{resumen.get('numeros_calientes_count', 0)} números calientes | "
                              f"{resumen.get('repeticiones_totales', 0)} repeticiones")
        
        elif opcion == '4':
            print("\n🔍 COMPARACIÓN ENTRE DÍAS:")
            dias_comparar = []
            for i, dia in enumerate(sistema.dias_semana, 1):
                print(f"  {i}. {dia.capitalize()}")
            
            try:
                dia1_idx = int(input("\nPrimer día a comparar: ")) - 1
                dia2_idx = int(input("Segundo día a comparar: ")) - 1
                
                if 0 <= dia1_idx < 7 and 0 <= dia2_idx < 7:
                    dia1 = sistema.dias_semana[dia1_idx]
                    dia2 = sistema.dias_semana[dia2_idx]
                    
                    # Aquí puedes añadir función de comparación
                    print(f"\nComparando {dia1} vs {dia2}...")
                    # sistema.comparar_dias(dia1, dia2)
                else:
                    print("❌ Opciones inválidas")
            except:
                print("❌ Entrada no válida")
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")
        
        # Preguntar si continuar
        continuar = input("\n¿Continuar? (s/n): ").strip().lower()
        if continuar != 's':
            print("\n👋 ¡Hasta luego!")
            break

if __name__ == "__main__":
    sistema_completo_avanzado()