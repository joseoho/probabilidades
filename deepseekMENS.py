import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Crear el DataFrame
data = {
    'hora': [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
    'lunes': [16, 4, 7, 1, 14, 30, 19, 26, 0, 30, 12, 28],
    'martes': [26, 15, 19, 0, 28, 9, 3, 20, 12, 8, 14, 20],
    'miércoles': [4, 5, 31, 36, 14, 21, 7, 32, 31, 29, 13, 21],
    'jueves': [25, 22, 15, 8, 4, 29, 16, 19, 1, 12, 12, 5],
    'viernes': [6, 30, 25, 19, 9, 33, 17, 0, 23, 31, 27, 5]
}

df = pd.DataFrame(data)

# Transformar los datos para que cada fila sea una observación
df_melted = df.melt(id_vars=['hora'], var_name='dia', value_name='numero')

# Convertir los días de la semana a números
df_melted['dia'] = df_melted['dia'].map({'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 'viernes': 4})

# Separar en variables independientes (X) y dependiente (y)
X = df_melted[['hora', 'dia']]
y = df_melted['numero']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')

# Predecir para el día 14/3/25 (viernes, dia=4)
horas = np.array([[hora, 4] for hora in range(8, 20)])  # Desde las 8 AM hasta las 7 PM
predicciones = model.predict(horas)

# Crear un DataFrame con las predicciones
predicciones_df = pd.DataFrame({
    'Hora': range(8, 20),
    'Predicción': predicciones
})

# Mostrar las predicciones en la consola
print(predicciones_df)

# Guardar las predicciones en un archivo de Excel
predicciones_df.to_excel('predicciones_14_3_25.xlsx', index=False)

print("Las predicciones se han guardado en 'predicciones_14_3_25.xlsx'.")