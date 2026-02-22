"""
4.1 Fundamentos de Inteligencia Artificial y Aprendizaje Automático
Tutorial: Entrenamiento de Modelo de Regresión Lineal
Dataset: California Housing Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

def cargar_datos():
    """
    Paso 1: Cargar el dataset de viviendas de California
    """
    print("=" * 70)
    print("PASO 1: CARGANDO DATOS")
    print("=" * 70)
    
    # Cargar el dataset
    housing = fetch_california_housing()
    
    # Crear un DataFrame para facilitar el análisis
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRECIO'] = housing.target
    
    print(f"\nDimensiones del dataset: {df.shape}")
    print(f"Número de muestras: {df.shape[0]}")
    print(f"Número de características: {df.shape[1] - 1}")
    
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())
    
    print("\nDescripción de las características:")
    for i, feature in enumerate(housing.feature_names):
        print(f"  - {feature}: {housing.feature_names[i]}")
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    return df, housing

def explorar_datos(df):
    """
    Paso 2: Exploración y visualización de datos
    """
    print("\n" + "=" * 70)
    print("PASO 2: EXPLORACIÓN DE DATOS")
    print("=" * 70)
    
    # Verificar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    # Matriz de correlación
    print("\nMatriz de correlación con el precio:")
    correlacion = df.corr()['PRECIO'].sort_values(ascending=False)
    print(correlacion)
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Distribución del precio
    axes[0, 0].hist(df['PRECIO'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Precio (en $100,000)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Precios de Viviendas')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Precio vs Ingreso Medio
    axes[0, 1].scatter(df['MedInc'], df['PRECIO'], alpha=0.3)
    axes[0, 1].set_xlabel('Ingreso Medio')
    axes[0, 1].set_ylabel('Precio (en $100,000)')
    axes[0, 1].set_title('Precio vs Ingreso Medio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Mapa de calor de correlación
    correlacion_matrix = df.corr()
    sns.heatmap(correlacion_matrix, annot=True, cmap='coolwarm', center=0,
                ax=axes[1, 0], fmt='.2f', square=True)
    axes[1, 0].set_title('Matriz de Correlación')
    
    # Gráfico 4: Precio vs Antigüedad de la casa
    axes[1, 1].scatter(df['HouseAge'], df['PRECIO'], alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Antigüedad de la Casa')
    axes[1, 1].set_ylabel('Precio (en $100,000)')
    axes[1, 1].set_title('Precio vs Antigüedad de la Casa')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_1_exploracion_datos.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos guardados en '4_1_exploracion_datos.png'")
    
    return correlacion

def preparar_datos(df):
    """
    Paso 3: Preparación de datos para el entrenamiento
    """
    print("\n" + "=" * 70)
    print("PASO 3: PREPARACIÓN DE DATOS")
    print("=" * 70)
    
    # Separar características (X) y variable objetivo (y)
    X = df.drop('PRECIO', axis=1)
    y = df['PRECIO']
    
    print(f"\nForma de X (características): {X.shape}")
    print(f"Forma de y (objetivo): {y.shape}")
    
    # Dividir en conjuntos de entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")
    print(f"Proporción: {X_train.shape[0]/X.shape[0]*100:.1f}% entrenamiento, "
          f"{X_test.shape[0]/X.shape[0]*100:.1f}% prueba")
    
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train):
    """
    Paso 4: Entrenar el modelo de Regresión Lineal
    """
    print("\n" + "=" * 70)
    print("PASO 4: ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    # Crear el modelo de regresión lineal
    modelo = LinearRegression()
    
    print("\nEntrenando el modelo de Regresión Lineal...")
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    print("✓ Modelo entrenado exitosamente!")
    
    # Mostrar los coeficientes
    print("\nCoeficientes del modelo:")
    for feature, coef in zip(X_train.columns, modelo.coef_):
        print(f"  {feature:12s}: {coef:10.4f}")
    
    print(f"\nIntercepción: {modelo.intercept_:.4f}")
    
    return modelo

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Paso 5: Evaluar el rendimiento del modelo
    """
    print("\n" + "=" * 70)
    print("PASO 5: EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    # Hacer predicciones
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    # Calcular métricas para el conjunto de entrenamiento
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calcular métricas para el conjunto de prueba
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n📊 MÉTRICAS DE RENDIMIENTO")
    print("-" * 70)
    print(f"{'Métrica':<25} {'Entrenamiento':>20} {'Prueba':>20}")
    print("-" * 70)
    print(f"{'MSE':<25} {train_mse:>20.4f} {test_mse:>20.4f}")
    print(f"{'RMSE':<25} {train_rmse:>20.4f} {test_rmse:>20.4f}")
    print(f"{'MAE':<25} {train_mae:>20.4f} {test_mae:>20.4f}")
    print(f"{'R² Score':<25} {train_r2:>20.4f} {test_r2:>20.4f}")
    print("-" * 70)
    
    print("\n📝 INTERPRETACIÓN:")
    print(f"  • El modelo explica el {test_r2*100:.2f}% de la variabilidad en los precios")
    print(f"  • Error promedio de predicción: ${test_mae*100000:.0f}")
    print(f"  • RMSE: ${test_rmse*100000:.0f}")
    
    # Visualizar resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1: Predicciones vs Valores reales
    axes[0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Predicción perfecta')
    axes[0].set_xlabel('Precio Real (en $100,000)')
    axes[0].set_ylabel('Precio Predicho (en $100,000)')
    axes[0].set_title('Predicciones vs Valores Reales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Distribución de residuos
    residuos = y_test - y_test_pred
    axes[1].hist(residuos, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuo (Error de Predicción)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Residuos')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_1_evaluacion_modelo.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos de evaluación guardados en '4_1_evaluacion_modelo.png'")
    
    return y_test_pred

def hacer_prediccion_ejemplo(modelo, X_test, y_test):
    """
    Paso 6: Realizar predicciones de ejemplo
    """
    print("\n" + "=" * 70)
    print("PASO 6: PREDICCIONES DE EJEMPLO")
    print("=" * 70)
    
    # Tomar 5 ejemplos aleatorios
    indices = np.random.choice(X_test.index, 5, replace=False)
    
    print("\nPredicciones de casas individuales:")
    print("-" * 70)
    
    for i, idx in enumerate(indices, 1):
        prediccion = modelo.predict(X_test.loc[[idx]])[0]
        real = y_test.loc[idx]
        error = abs(prediccion - real)
        
        print(f"\nEjemplo {i}:")
        print(f"  Características:")
        for col in X_test.columns:
            print(f"    {col:12s}: {X_test.loc[idx, col]:.2f}")
        print(f"  Precio Real:      ${real*100000:,.0f}")
        print(f"  Precio Predicho:  ${prediccion*100000:,.0f}")
        print(f"  Error:            ${error*100000:,.0f} ({error/real*100:.1f}%)")

def main():
    """
    Función principal que ejecuta todo el tutorial
    """
    print("\n" + "=" * 70)
    print("TUTORIAL: REGRESIÓN LINEAL CON SCIKIT-LEARN")
    print("Dataset: California Housing")
    print("=" * 70)
    
    # Ejecutar todos los pasos
    df, housing = cargar_datos()
    correlacion = explorar_datos(df)
    X_train, X_test, y_train, y_test = preparar_datos(df)
    modelo = entrenar_modelo(X_train, y_train)
    y_pred = evaluar_modelo(modelo, X_train, X_test, y_train, y_test)
    hacer_prediccion_ejemplo(modelo, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("✅ TUTORIAL COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  • 4_1_exploracion_datos.png - Visualización de datos")
    print("  • 4_1_evaluacion_modelo.png - Evaluación del modelo")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
