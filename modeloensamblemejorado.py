import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

# Leer datos desde el archivo Excel
df_historico = pd.read_excel('animalitos01012324.xlsx')

# Verificar los tipos de datos
print(df_historico.dtypes)

# Asegurarse de que las columnas relevantes sean numéricas
for col in ['lunes', 'martes', 'miercoles', 'jueves', 'sábado','domingo']:
    df_historico[col] = pd.to_numeric(df_historico[col], errors='coerce')

# Separar características y objetivo
X_historico = df_historico[['lunes', 'martes', 'miercoles', 'jueves', 'sábado','domingo']]
y_train = df_historico[['lunes', 'martes', 'miercoles', 'jueves', 'sábado','domingo']].mean(axis=1, skipna=True)

# Filtrar filas donde y_train no sea NaN
mask = ~y_train.isna()
X_train = X_historico[mask]
y_train = y_train[mask]

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train_final, X_test, y_train_final, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definir los modelos base
model_rf = RandomForestRegressor(random_state=42)
model_gb = GradientBoostingRegressor(random_state=42)
model_xgb = XGBRegressor(random_state=42)

# Crear el Voting Regressor
voting_model = VotingRegressor(estimators=[('rf', model_rf), ('gb', model_gb), ('xgb', model_xgb)])

# Crear el Stacking Regressor
stacking_model = StackingRegressor(
    estimators=[('rf', model_rf), ('gb', model_gb), ('xgb', model_xgb)],
    final_estimator=XGBRegressor(random_state=42)
)

# Entrenar el modelo Voting Regressor
voting_model.fit(X_train_final, y_train_final)

# Predecir para el conjunto de prueba
predicciones_viernes_voting = voting_model.predict(X_test)

# Calcular métricas de error para Voting Regressor
mse_voting = mean_squared_error(y_test, predicciones_viernes_voting)
rmse_voting = np.sqrt(mse_voting)
mae_voting = mean_absolute_error(y_test, predicciones_viernes_voting)

print(f'Voting Regressor - Mean Squared Error (MSE): {mse_voting}')
print(f'Voting Regressor - Root Mean Squared Error (RMSE): {rmse_voting}')
print(f'Voting Regressor - Mean Absolute Error (MAE): {mae_voting}')

# Entrenar el modelo Stacking Regressor
stacking_model.fit(X_train_final, y_train_final)

# Predecir para el conjunto de prueba
predicciones_viernes_stacking = stacking_model.predict(X_test)

# Calcular métricas de error para Stacking Regressor
mse_stacking = mean_squared_error(y_test, predicciones_viernes_stacking)
rmse_stacking = np.sqrt(mse_stacking)
mae_stacking = mean_absolute_error(y_test, predicciones_viernes_stacking)

print(f'Stacking Regressor - Mean Squared Error (MSE): {mse_stacking}')
print(f'Stacking Regressor - Root Mean Squared Error (RMSE): {rmse_stacking}')
print(f'Stacking Regressor - Mean Absolute Error (MAE): {mae_stacking}')

# Agregar las predicciones del día al DataFrame en la columna correspondiente
df_historico.loc[X_test.index, 'viernes'] = predicciones_viernes_stacking

# Mostrar resultados organizados por hora
resultados_miercoles = df_historico[['hora', 'viernes']].sort_values(by='hora')

# Guardar resultados en un nuevo archivo Excel
resultados_miercoles.to_excel('predicciones_viernes_stacking.xlsx', index=False)

print("Las predicciones se han guardado en el archivo 'predicciones_viernes_stacking.xlsx'.")
