"""
4.2 Aplicaciones y Casos de Uso de IA y Machine Learning
Sector: SALUD
Aplicación: Sistema de Predicción de Riesgo de Diabetes
Dataset: Pima Indians Diabetes Database

PROBLEMA REAL: 
Predecir si un paciente tiene riesgo de desarrollar diabetes basándose en 
mediciones diagnósticas, permitiendo intervención temprana y tratamiento preventivo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, accuracy_score, 
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

class SistemaDiagnosticoDiabetes:
    """
    Sistema de predicción de riesgo de diabetes usando Machine Learning
    """
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.feature_names = None
        
    def paso1_recoleccion_datos(self):
        """
        PASO 1: RECOLECCIÓN Y CARGA DE DATOS
        """
        print("=" * 80)
        print("PASO 1: RECOLECCIÓN Y CARGA DE DATOS")
        print("=" * 80)
        
        # Cargar dataset de diabetes
        # Características del dataset:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        columnas = [
            'Embarazos',
            'Glucosa',
            'PresionSanguinea',
            'GrosorPiel',
            'Insulina',
            'IMC',
            'FuncionPedigriDiabetes',
            'Edad',
            'Resultado'
        ]
        
        try:
            df = pd.read_csv(url, names=columnas)
            print("✓ Datos cargados desde repositorio en línea")
        except:
            # Datos de ejemplo si no hay conexión
            print("⚠ Creando datos de ejemplo (sin conexión)")
            np.random.seed(42)
            n_samples = 768
            df = pd.DataFrame({
                'Embarazos': np.random.randint(0, 17, n_samples),
                'Glucosa': np.random.randint(0, 200, n_samples),
                'PresionSanguinea': np.random.randint(0, 122, n_samples),
                'GrosorPiel': np.random.randint(0, 100, n_samples),
                'Insulina': np.random.randint(0, 846, n_samples),
                'IMC': np.random.uniform(0, 67, n_samples),
                'FuncionPedigriDiabetes': np.random.uniform(0.078, 2.42, n_samples),
                'Edad': np.random.randint(21, 81, n_samples),
                'Resultado': np.random.randint(0, 2, n_samples)
            })
        
        print(f"\n📊 Información del Dataset:")
        print(f"  • Total de pacientes: {len(df)}")
        print(f"  • Número de características: {len(df.columns) - 1}")
        print(f"  • Pacientes con diabetes: {df['Resultado'].sum()} ({df['Resultado'].sum()/len(df)*100:.1f}%)")
        print(f"  • Pacientes sin diabetes: {len(df) - df['Resultado'].sum()} ({(len(df) - df['Resultado'].sum())/len(df)*100:.1f}%)")
        
        print(f"\n📋 Descripción de las características:")
        descripciones = {
            'Embarazos': 'Número de embarazos',
            'Glucosa': 'Concentración de glucosa en plasma (mg/dl)',
            'PresionSanguinea': 'Presión arterial diastólica (mm Hg)',
            'GrosorPiel': 'Grosor del pliegue cutáneo del tríceps (mm)',
            'Insulina': 'Insulina sérica de 2 horas (mu U/ml)',
            'IMC': 'Índice de masa corporal (peso en kg/(altura en m)²)',
            'FuncionPedigriDiabetes': 'Función de pedigrí de diabetes',
            'Edad': 'Edad en años',
            'Resultado': 'Variable objetivo (0=No diabetes, 1=Diabetes)'
        }
        
        for col, desc in descripciones.items():
            print(f"  • {col:25s}: {desc}")
        
        print("\n📈 Primeras 5 filas del dataset:")
        print(df.head())
        
        return df
    
    def paso2_analisis_exploratorio(self, df):
        """
        PASO 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
        """
        print("\n" + "=" * 80)
        print("PASO 2: ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 80)
        
        # Estadísticas descriptivas
        print("\n📊 Estadísticas descriptivas:")
        print(df.describe())
        
        # Verificar valores faltantes
        print("\n🔍 Valores faltantes:")
        print(df.isnull().sum())
        
        # Detectar valores cero que podrían ser faltantes
        print("\n⚠️  Valores cero (potencialmente faltantes):")
        columnas_con_ceros = ['Glucosa', 'PresionSanguinea', 'GrosorPiel', 'Insulina', 'IMC']
        for col in columnas_con_ceros:
            ceros = (df[col] == 0).sum()
            if ceros > 0:
                print(f"  • {col:25s}: {ceros} valores ({ceros/len(df)*100:.1f}%)")
        
        # Crear visualizaciones
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Distribución de clases
        ax1 = plt.subplot(3, 3, 1)
        df['Resultado'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
        plt.title('Distribución de Diagnósticos', fontweight='bold')
        plt.xlabel('Diagnóstico')
        plt.ylabel('Número de Pacientes')
        plt.xticks([0, 1], ['Sin Diabetes', 'Con Diabetes'], rotation=0)
        
        # 2-9. Histogramas de características
        caracteristicas = [col for col in df.columns if col != 'Resultado']
        for idx, col in enumerate(caracteristicas, 2):
            ax = plt.subplot(3, 3, idx)
            df[df['Resultado']==0][col].hist(alpha=0.5, bins=20, label='Sin Diabetes', color='#2ecc71')
            df[df['Resultado']==1][col].hist(alpha=0.5, bins=20, label='Con Diabetes', color='#e74c3c')
            plt.title(f'Distribución: {col}', fontsize=10, fontweight='bold')
            plt.xlabel(col, fontsize=8)
            plt.ylabel('Frecuencia', fontsize=8)
            plt.legend(fontsize=7)
        
        plt.tight_layout()
        plt.savefig('4_2_exploracion_datos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico guardado: '4_2_exploracion_datos.png'")
        
        # Matriz de correlación
        plt.figure(figsize=(12, 10))
        correlacion = df.corr()
        sns.heatmap(correlacion, annot=True, cmap='RdYlGn', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Matriz de Correlación de Características', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('4_2_correlacion.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado: '4_2_correlacion.png'")
        
        # Análisis de correlación con el resultado
        print("\n🔗 Correlación con el diagnóstico de diabetes:")
        corr_resultado = correlacion['Resultado'].sort_values(ascending=False)
        for feature, corr_value in corr_resultado.items():
            if feature != 'Resultado':
                print(f"  • {feature:25s}: {corr_value:>6.3f}")
        
        return df
    
    def paso3_preprocesamiento(self, df):
        """
        PASO 3: PREPROCESAMIENTO Y LIMPIEZA DE DATOS
        """
        print("\n" + "=" * 80)
        print("PASO 3: PREPROCESAMIENTO Y LIMPIEZA DE DATOS")
        print("=" * 80)
        
        df_limpio = df.copy()
        
        # Reemplazar ceros por la mediana en columnas donde cero no es válido
        columnas_con_ceros = ['Glucosa', 'PresionSanguinea', 'GrosorPiel', 'Insulina', 'IMC']
        
        print("\n🔧 Reemplazando valores cero con la mediana:")
        for col in columnas_con_ceros:
            if (df_limpio[col] == 0).sum() > 0:
                mediana = df_limpio[df_limpio[col] != 0][col].median()
                ceros_antes = (df_limpio[col] == 0).sum()
                df_limpio[col] = df_limpio[col].replace(0, mediana)
                print(f"  • {col:25s}: {ceros_antes} valores reemplazados con {mediana:.2f}")
        
        # Separar características y variable objetivo
        X = df_limpio.drop('Resultado', axis=1)
        y = df_limpio['Resultado']
        
        self.feature_names = X.columns.tolist()
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📦 División de datos:")
        print(f"  • Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  • Prueba:        {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
        
        # Normalizar características
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n✓ Datos normalizados usando StandardScaler")
        print(f"  Media de características después de normalización: {X_train_scaled.mean():.6f}")
        print(f"  Desviación estándar después de normalización: {X_train_scaled.std():.6f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def paso4_entrenamiento(self, X_train, y_train):
        """
        PASO 4: ENTRENAMIENTO DEL MODELO
        """
        print("\n" + "=" * 80)
        print("PASO 4: ENTRENAMIENTO DEL MODELO")
        print("=" * 80)
        
        print("\n🤖 Modelo seleccionado: Random Forest Classifier")
        print("   Razones:")
        print("   • Maneja bien características no lineales")
        print("   • Resistente al sobreajuste")
        print("   • Proporciona importancia de características")
        print("   • Buen rendimiento en problemas médicos")
        
        # Crear y entrenar el modelo
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n⚙️  Hiperparámetros:")
        print(f"   • Número de árboles: 100")
        print(f"   • Profundidad máxima: 10")
        print(f"   • Muestras mínimas para dividir: 5")
        print(f"   • Muestras mínimas por hoja: 2")
        
        print("\n🔄 Entrenando modelo...")
        self.modelo.fit(X_train, y_train)
        print("✓ Modelo entrenado exitosamente!")
        
        # Validación cruzada
        print("\n📊 Validación cruzada (5-fold):")
        cv_scores = cross_val_score(self.modelo, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Precisión promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.modelo
    
    def paso5_evaluacion(self, X_train, X_test, y_train, y_test):
        """
        PASO 5: EVALUACIÓN DEL MODELO
        """
        print("\n" + "=" * 80)
        print("PASO 5: EVALUACIÓN DEL MODELO")
        print("=" * 80)
        
        # Predicciones
        y_train_pred = self.modelo.predict(X_train)
        y_test_pred = self.modelo.predict(X_test)
        y_test_proba = self.modelo.predict_proba(X_test)[:, 1]
        
        # Métricas
        print("\n📊 MÉTRICAS DE RENDIMIENTO:")
        print("-" * 80)
        print(f"{'Métrica':<30} {'Entrenamiento':>20} {'Prueba':>20}")
        print("-" * 80)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"{'Exactitud (Accuracy)':<30} {train_acc:>20.4f} {test_acc:>20.4f}")
        
        train_prec = precision_score(y_train, y_train_pred)
        test_prec = precision_score(y_test, y_test_pred)
        print(f"{'Precisión (Precision)':<30} {train_prec:>20.4f} {test_prec:>20.4f}")
        
        train_rec = recall_score(y_train, y_train_pred)
        test_rec = recall_score(y_test, y_test_pred)
        print(f"{'Sensibilidad (Recall)':<30} {train_rec:>20.4f} {test_rec:>20.4f}")
        
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        print(f"{'F1-Score':<30} {train_f1:>20.4f} {test_f1:>20.4f}")
        print("-" * 80)
        
        # Reporte de clasificación detallado
        print("\n📋 REPORTE DE CLASIFICACIÓN DETALLADO:")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Sin Diabetes', 'Con Diabetes']))
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matriz de confusión
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Sin Diabetes', 'Con Diabetes'],
                   yticklabels=['Sin Diabetes', 'Con Diabetes'])
        axes[0, 0].set_title('Matriz de Confusión', fontweight='bold')
        axes[0, 0].set_ylabel('Valor Real')
        axes[0, 0].set_xlabel('Valor Predicho')
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Tasa de Falsos Positivos')
        axes[0, 1].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0, 1].set_title('Curva ROC', fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Importancia de características
        importancias = self.modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]
        axes[1, 0].barh(range(len(importancias)), importancias[indices], color='steelblue')
        axes[1, 0].set_yticks(range(len(importancias)))
        axes[1, 0].set_yticklabels([self.feature_names[i] for i in indices])
        axes[1, 0].set_xlabel('Importancia')
        axes[1, 0].set_title('Importancia de Características', fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # 4. Distribución de probabilidades predichas
        axes[1, 1].hist(y_test_proba[y_test == 0], bins=20, alpha=0.5, 
                       label='Sin Diabetes', color='green')
        axes[1, 1].hist(y_test_proba[y_test == 1], bins=20, alpha=0.5, 
                       label='Con Diabetes', color='red')
        axes[1, 1].set_xlabel('Probabilidad Predicha')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Probabilidades', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('4_2_evaluacion_modelo.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficos guardados: '4_2_evaluacion_modelo.png'")
        
        # Interpretación clínica
        print("\n🏥 INTERPRETACIÓN CLÍNICA:")
        print(f"  • De cada 100 pacientes con diabetes, el modelo detecta {test_rec*100:.0f}")
        print(f"  • De cada 100 diagnósticos positivos, {test_prec*100:.0f} son correctos")
        print(f"  • Área bajo la curva ROC: {roc_auc:.3f} ({'Excelente' if roc_auc > 0.9 else 'Bueno' if roc_auc > 0.8 else 'Aceptable'})")
        
        print("\n💡 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i in range(min(5, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {self.feature_names[idx]:<30s}: {importancias[idx]:.4f}")
        
        return y_test_pred, y_test_proba
    
    def paso6_prediccion_paciente(self, X_test_original, y_test):
        """
        PASO 6: PREDICCIÓN PARA PACIENTES INDIVIDUALES
        """
        print("\n" + "=" * 80)
        print("PASO 6: SISTEMA DE PREDICCIÓN PARA PACIENTES")
        print("=" * 80)
        
        # Seleccionar algunos pacientes de ejemplo
        indices = [0, 10, 20, 30, 40]
        
        print("\n🏥 EJEMPLOS DE DIAGNÓSTICO:")
        print("=" * 80)
        
        for i, idx in enumerate(indices, 1):
            paciente = X_test_original.iloc[idx:idx+1]
            paciente_scaled = self.scaler.transform(paciente)
            
            prediccion = self.modelo.predict(paciente_scaled)[0]
            probabilidad = self.modelo.predict_proba(paciente_scaled)[0]
            real = y_test.iloc[idx]
            
            print(f"\n📋 PACIENTE #{i}")
            print("-" * 80)
            print("Datos clínicos:")
            for col in paciente.columns:
                print(f"  • {col:25s}: {paciente[col].values[0]:.2f}")
            
            print(f"\n🎯 Diagnóstico:")
            print(f"  • Probabilidad Sin Diabetes: {probabilidad[0]*100:.1f}%")
            print(f"  • Probabilidad Con Diabetes: {probabilidad[1]*100:.1f}%")
            print(f"  • Predicción del modelo:     {'CON DIABETES ⚠️' if prediccion == 1 else 'SIN DIABETES ✓'}")
            print(f"  • Diagnóstico real:          {'CON DIABETES' if real == 1 else 'SIN DIABETES'}")
            print(f"  • Resultado:                 {'✓ CORRECTO' if prediccion == real else '✗ INCORRECTO'}")
            
            # Nivel de riesgo
            prob_diabetes = probabilidad[1]
            if prob_diabetes < 0.3:
                riesgo = "BAJO"
                color = "🟢"
            elif prob_diabetes < 0.7:
                riesgo = "MODERADO"
                color = "🟡"
            else:
                riesgo = "ALTO"
                color = "🔴"
            
            print(f"  • Nivel de riesgo:           {color} {riesgo}")

def main():
    """
    Función principal que ejecuta el sistema completo
    """
    print("\n" + "=" * 80)
    print("SISTEMA DE PREDICCIÓN DE RIESGO DE DIABETES")
    print("Aplicación de Machine Learning en el Sector Salud")
    print("=" * 80)
    
    # Crear instancia del sistema
    sistema = SistemaDiagnosticoDiabetes()
    
    # Ejecutar todos los pasos del proceso
    df = sistema.paso1_recoleccion_datos()
    df = sistema.paso2_analisis_exploratorio(df)
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = sistema.paso3_preprocesamiento(df)
    modelo = sistema.paso4_entrenamiento(X_train, y_train)
    y_pred, y_proba = sistema.paso5_evaluacion(X_train, X_test, y_train, y_test)
    sistema.paso6_prediccion_paciente(X_test_orig, y_test)
    
    print("\n" + "=" * 80)
    print("✅ SISTEMA COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print("\n📁 Archivos generados:")
    print("  • 4_2_exploracion_datos.png - Análisis exploratorio")
    print("  • 4_2_correlacion.png - Matriz de correlación")
    print("  • 4_2_evaluacion_modelo.png - Métricas y visualizaciones")
    print("\n💡 Este sistema puede integrarse en aplicaciones clínicas reales")
    print("   para asistir en el diagnóstico temprano de diabetes.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
