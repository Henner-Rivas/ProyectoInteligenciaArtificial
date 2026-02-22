"""
4.4 Herramientas y Frameworks de IA y Machine Learning
Framework seleccionado: SCIKIT-LEARN

JUSTIFICACIÓN:
- Biblioteca más popular de ML en Python
- APIs consistentes y bien documentadas
- Amplia variedad de algoritmos implementados
- Excelente integración con el ecosistema científico de Python
- Ideal para prototipado rápido y producción

PROYECTO: Sistema de Reconocimiento de Dígitos Escritos a Mano
Dataset: MNIST (dígitos del 0-9)
Modelo: Red Neuronal Multi-capa (MLPClassifier)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.decomposition import PCA
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class SistemaReconocimientoDigitos:
    """
    Sistema de reconocimiento de dígitos usando Scikit-learn
    """
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.mejor_modelo = None
        self.historial_entrenamiento = {}
        
    def presentacion_framework(self):
        """
        Presentar el framework Scikit-learn
        """
        print("=" * 80)
        print("FRAMEWORK: SCIKIT-LEARN")
        print("=" * 80)
        
        print("\n📚 ¿QUÉ ES SCIKIT-LEARN?")
        print("-" * 80)
        print("Scikit-learn es una biblioteca de aprendizaje automático de código abierto")
        print("para Python. Construida sobre NumPy, SciPy y Matplotlib.")
        
        print("\n✨ CARACTERÍSTICAS PRINCIPALES:")
        print("  1. Simple y eficiente - API consistente y fácil de usar")
        print("  2. Completa - Clasificación, regresión, clustering, reducción")
        print("  3. Bien documentada - Excelentes tutoriales y ejemplos")
        print("  4. Producción lista - Código robusto y optimizado")
        print("  5. Comunidad activa - Amplio soporte y desarrollo continuo")
        
        print("\n🔧 COMPONENTES PRINCIPALES:")
        print("  • Preprocesamiento: StandardScaler, MinMaxScaler, PCA")
        print("  • Clasificación: SVM, Random Forest, Neural Networks")
        print("  • Regresión: Linear, Ridge, Lasso, ElasticNet")
        print("  • Clustering: K-Means, DBSCAN, Hierarchical")
        print("  • Validación: cross_val_score, GridSearchCV")
        print("  • Métricas: accuracy, precision, recall, f1-score")
        
        print("\n🎯 ¿POR QUÉ SCIKIT-LEARN PARA ESTE PROYECTO?")
        print("  ✓ MLPClassifier incorporado para redes neuronales")
        print("  ✓ Herramientas de validación y optimización integradas")
        print("  ✓ Fácil exportación de modelos con joblib")
        print("  ✓ Excelente para prototipado rápido")
        print("  ✓ Rendimiento competitivo para datasets pequeños/medianos")
        
        print("\n🆚 COMPARACIÓN CON OTROS FRAMEWORKS:")
        print("-" * 80)
        comparacion = pd.DataFrame({
            'Framework': ['Scikit-learn', 'TensorFlow', 'PyTorch'],
            'Facilidad de uso': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐'],
            'Deep Learning': ['⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'ML Tradicional': ['⭐⭐⭐⭐⭐', '⭐⭐', '⭐⭐'],
            'Producción': ['⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐'],
            'Documentación': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐']
        })
        print(comparacion.to_string(index=False))
        print("-" * 80)
    
    def cargar_y_explorar_datos(self):
        """
        Paso 1: Cargar y explorar el dataset de dígitos
        """
        print("\n" + "=" * 80)
        print("PASO 1: CARGA Y EXPLORACIÓN DE DATOS")
        print("=" * 80)
        
        # Cargar dataset de dígitos (versión pequeña de MNIST)
        digits = load_digits()
        
        print("\n📊 Información del Dataset:")
        print(f"  • Nombre: Optical Recognition of Handwritten Digits")
        print(f"  • Total de imágenes: {len(digits.data)}")
        print(f"  • Dimensión de cada imagen: 8x8 píxeles")
        print(f"  • Total de características: {digits.data.shape[1]} (64 píxeles)")
        print(f"  • Clases (dígitos): {len(digits.target_names)} ({digits.target_names})")
        print(f"  • Rango de valores: 0-16 (escala de grises)")
        
        # Distribución de clases
        print("\n📈 Distribución de clases:")
        unique, counts = np.unique(digits.target, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"  • Dígito {digit}: {count} imágenes ({count/len(digits.target)*100:.1f}%)")
        
        # Visualizar ejemplos
        fig, axes = plt.subplots(4, 10, figsize=(15, 6))
        fig.suptitle('Ejemplos de Dígitos del Dataset', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            if i < 40:
                ax.imshow(digits.images[i], cmap='gray')
                ax.set_title(f'Dígito: {digits.target[i]}', fontsize=9)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('4_4_dataset_ejemplos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualización guardada: '4_4_dataset_ejemplos.png'")
        
        # Analizar variabilidad
        print("\n🔍 Estadísticas de píxeles:")
        print(f"  • Media: {digits.data.mean():.2f}")
        print(f"  • Desviación estándar: {digits.data.std():.2f}")
        print(f"  • Mínimo: {digits.data.min():.2f}")
        print(f"  • Máximo: {digits.data.max():.2f}")
        
        return digits
    
    def preprocesamiento_datos(self, digits):
        """
        Paso 2: Preprocesamiento y preparación de datos
        """
        print("\n" + "=" * 80)
        print("PASO 2: PREPROCESAMIENTO DE DATOS")
        print("=" * 80)
        
        X = digits.data
        y = digits.target
        
        print("\n🔧 Operaciones de preprocesamiento:")
        print("  1. División train/test (80/20)")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"     ✓ Entrenamiento: {len(X_train)} muestras")
        print(f"     ✓ Prueba: {len(X_test)} muestras")
        
        # Normalizar datos
        print("  2. Normalización (StandardScaler)")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"     ✓ Media después de normalización: {X_train_scaled.mean():.6f}")
        print(f"     ✓ Std después de normalización: {X_train_scaled.std():.6f}")
        
        # Análisis con PCA (opcional, para visualización)
        print("  3. Análisis de Componentes Principales (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_scaled)
        
        print(f"     ✓ Varianza explicada por PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"     ✓ Varianza explicada por PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
        print(f"     ✓ Varianza total explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        # Visualizar PCA
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, 
                            cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Dígito')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
        plt.title('Visualización PCA del Dataset de Dígitos', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('4_4_pca_visualizacion.png', dpi=300, bbox_inches='tight')
        print("     ✓ Visualización PCA guardada: '4_4_pca_visualizacion.png'")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def disenar_modelo(self):
        """
        Paso 3: Diseño y configuración del modelo
        """
        print("\n" + "=" * 80)
        print("PASO 3: DISEÑO DEL MODELO")
        print("=" * 80)
        
        print("\n🧠 Modelo seleccionado: Multi-Layer Perceptron (MLP)")
        print("\n📐 Arquitectura de la red neuronal:")
        print("  • Capa de entrada:  64 neuronas (8x8 píxeles)")
        print("  • Capa oculta 1:    128 neuronas (ReLU)")
        print("  • Capa oculta 2:    64 neuronas (ReLU)")
        print("  • Capa de salida:   10 neuronas (10 dígitos)")
        
        print("\n⚙️  Hiperparámetros:")
        print("  • Función de activación: ReLU")
        print("  • Solver: adam (optimizador adaptativo)")
        print("  • Learning rate: adaptativo")
        print("  • Max iteraciones: 500")
        print("  • Batch size: auto")
        print("  • Early stopping: Sí")
        
        # Crear modelo
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        
        print("\n✓ Modelo configurado correctamente")
        
        return self.modelo
    
    def entrenar_modelo(self, X_train, y_train):
        """
        Paso 4: Entrenamiento del modelo
        """
        print("\n" + "=" * 80)
        print("PASO 4: ENTRENAMIENTO DEL MODELO")
        print("=" * 80)
        
        print("\n🔄 Iniciando entrenamiento...")
        inicio = time.time()
        
        self.modelo.fit(X_train, y_train)
        
        tiempo_entrenamiento = time.time() - inicio
        
        print(f"✓ Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
        print(f"\n📊 Detalles del entrenamiento:")
        print(f"  • Iteraciones realizadas: {self.modelo.n_iter_}")
        print(f"  • Función de pérdida final: {self.modelo.loss_:.6f}")
        print(f"  • Número de capas: {self.modelo.n_layers_}")
        print(f"  • Número de salidas: {self.modelo.n_outputs_}")
        
        # Guardar historial
        self.historial_entrenamiento['tiempo'] = tiempo_entrenamiento
        self.historial_entrenamiento['iteraciones'] = self.modelo.n_iter_
        self.historial_entrenamiento['loss_final'] = self.modelo.loss_
        
        # Curva de pérdida
        if hasattr(self.modelo, 'loss_curve_'):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.modelo.loss_curve_, linewidth=2)
            plt.xlabel('Iteraciones')
            plt.ylabel('Pérdida')
            plt.title('Curva de Pérdida durante Entrenamiento', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if hasattr(self.modelo, 'validation_scores_'):
                plt.plot(self.modelo.validation_scores_, linewidth=2, color='green')
                plt.xlabel('Iteraciones')
                plt.ylabel('Score de Validación')
                plt.title('Score de Validación durante Entrenamiento', fontweight='bold')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('4_4_entrenamiento_curvas.png', dpi=300, bbox_inches='tight')
            print("  ✓ Curvas de entrenamiento guardadas: '4_4_entrenamiento_curvas.png'")
    
    def optimizar_hiperparametros(self, X_train, y_train):
        """
        Paso 5: Optimización de hiperparámetros con GridSearchCV
        """
        print("\n" + "=" * 80)
        print("PASO 5: OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearchCV)")
        print("=" * 80)
        
        print("\n🔍 Buscando los mejores hiperparámetros...")
        print("   (Esto puede tomar algunos minutos)")
        
        # Definir espacio de búsqueda
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (100, 50)],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01]
        }
        
        print("\n📋 Espacio de búsqueda:")
        for param, valores in param_grid.items():
            print(f"  • {param}: {valores}")
        
        # Crear GridSearchCV
        grid_search = GridSearchCV(
            MLPClassifier(max_iter=300, random_state=42, early_stopping=True),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        inicio = time.time()
        grid_search.fit(X_train, y_train)
        tiempo_busqueda = time.time() - inicio
        
        print(f"\n✓ Búsqueda completada en {tiempo_busqueda:.2f} segundos")
        print(f"\n🏆 Mejores hiperparámetros encontrados:")
        for param, valor in grid_search.best_params_.items():
            print(f"  • {param}: {valor}")
        
        print(f"\n📊 Mejor score de validación cruzada: {grid_search.best_score_:.4f}")
        
        # Guardar mejor modelo
        self.mejor_modelo = grid_search.best_estimator_
        
        # Mostrar resultados de todas las combinaciones
        resultados = pd.DataFrame(grid_search.cv_results_)
        print("\n📈 Top 5 configuraciones:")
        top_5 = resultados.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]
        print(top_5.to_string(index=False))
        
        return self.mejor_modelo
    
    def evaluar_modelo(self, X_test, y_test):
        """
        Paso 6: Evaluación exhaustiva del modelo
        """
        print("\n" + "=" * 80)
        print("PASO 6: EVALUACIÓN DEL MODELO")
        print("=" * 80)
        
        # Usar el mejor modelo si existe, sino usar el modelo básico
        modelo_evaluar = self.mejor_modelo if self.mejor_modelo else self.modelo
        
        print("\n🎯 Realizando predicciones...")
        inicio = time.time()
        y_pred = modelo_evaluar.predict(X_test)
        tiempo_prediccion = time.time() - inicio
        
        print(f"✓ Predicciones completadas en {tiempo_prediccion*1000:.2f} ms")
        print(f"  • Velocidad: {len(X_test)/tiempo_prediccion:.0f} predicciones/segundo")
        
        # Métricas generales
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print("\n📊 MÉTRICAS GLOBALES:")
        print("-" * 80)
        print(f"  • Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  • Precision (weighted):  {precision:.4f}")
        print(f"  • Recall (weighted):     {recall:.4f}")
        print(f"  • F1-Score (weighted):   {f1:.4f}")
        print("-" * 80)
        
        # Reporte detallado por clase
        print("\n📋 REPORTE DETALLADO POR DÍGITO:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[f'Dígito {i}' for i in range(10)]))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        
        # 1. Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=range(10), yticklabels=range(10))
        axes[0, 0].set_title('Matriz de Confusión', fontweight='bold', fontsize=14)
        axes[0, 0].set_ylabel('Valor Real')
        axes[0, 0].set_xlabel('Valor Predicho')
        
        # 2. Accuracy por dígito
        accuracy_por_digito = cm.diagonal() / cm.sum(axis=1)
        axes[0, 1].bar(range(10), accuracy_por_digito, color='steelblue')
        axes[0, 1].set_xlabel('Dígito')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy por Dígito', fontweight='bold', fontsize=14)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_xticks(range(10))
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracy_por_digito):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # 3. Errores por dígito
        errores_por_digito = cm.sum(axis=1) - cm.diagonal()
        axes[1, 0].bar(range(10), errores_por_digito, color='coral')
        axes[1, 0].set_xlabel('Dígito')
        axes[1, 0].set_ylabel('Número de Errores')
        axes[1, 0].set_title('Errores por Dígito', fontweight='bold', fontsize=14)
        axes[1, 0].set_xticks(range(10))
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Matriz de confusión normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 1],
                   xticklabels=range(10), yticklabels=range(10), vmin=0, vmax=1)
        axes[1, 1].set_title('Matriz de Confusión Normalizada', fontweight='bold', fontsize=14)
        axes[1, 1].set_ylabel('Valor Real')
        axes[1, 1].set_xlabel('Valor Predicho')
        
        plt.tight_layout()
        plt.savefig('4_4_evaluacion_modelo.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualización guardada: '4_4_evaluacion_modelo.png'")
        
        # Análisis de errores
        print("\n🔍 ANÁLISIS DE ERRORES:")
        errores_mask = y_test != y_pred
        num_errores = errores_mask.sum()
        print(f"  • Total de errores: {num_errores} de {len(y_test)} ({num_errores/len(y_test)*100:.2f}%)")
        
        if num_errores > 0:
            print("\n  Confusiones más comunes:")
            confusiones = []
            for i in range(10):
                for j in range(10):
                    if i != j and cm[i, j] > 0:
                        confusiones.append((i, j, cm[i, j]))
            confusiones.sort(key=lambda x: x[2], reverse=True)
            for real, pred, count in confusiones[:5]:
                print(f"    • Dígito {real} confundido con {pred}: {count} veces")
        
        return y_pred, accuracy
    
    def guardar_modelo(self):
        """
        Paso 7: Guardar el modelo entrenado
        """
        print("\n" + "=" * 80)
        print("PASO 7: GUARDAR MODELO")
        print("=" * 80)
        
        modelo_guardar = self.mejor_modelo if self.mejor_modelo else self.modelo
        
        # Guardar modelo
        joblib.dump(modelo_guardar, '4_4_modelo_digitos.pkl')
        joblib.dump(self.scaler, '4_4_scaler.pkl')
        
        print("\n💾 Modelos guardados:")
        print("  • 4_4_modelo_digitos.pkl - Modelo entrenado")
        print("  • 4_4_scaler.pkl - Escalador de datos")
        
        import os
        size_modelo = os.path.getsize('4_4_modelo_digitos.pkl')
        size_scaler = os.path.getsize('4_4_scaler.pkl')
        
        print(f"\n📦 Tamaño de archivos:")
        print(f"  • Modelo: {size_modelo/1024:.2f} KB")
        print(f"  • Scaler: {size_scaler/1024:.2f} KB")
        print(f"  • Total: {(size_modelo + size_scaler)/1024:.2f} KB")
        
        print("\n💡 Uso del modelo guardado:")
        print("   # Cargar modelo")
        print("   modelo = joblib.load('4_4_modelo_digitos.pkl')")
        print("   scaler = joblib.load('4_4_scaler.pkl')")
        print("   # Hacer predicción")
        print("   X_nuevo_scaled = scaler.transform(X_nuevo)")
        print("   prediccion = modelo.predict(X_nuevo_scaled)")
    
    def demostrar_predicciones(self, X_test_original, X_test_scaled, y_test):
        """
        Paso 8: Demostración de predicciones en tiempo real
        """
        print("\n" + "=" * 80)
        print("PASO 8: DEMOSTRACIÓN DE PREDICCIONES")
        print("=" * 80)
        
        modelo_usar = self.mejor_modelo if self.mejor_modelo else self.modelo
        
        # Seleccionar ejemplos aleatorios
        indices = np.random.choice(len(X_test_scaled), 20, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle('Predicciones del Modelo', fontsize=16, fontweight='bold')
        
        correctas = 0
        incorrectas = 0
        
        for idx, ax in enumerate(axes.flat):
            i = indices[idx]
            imagen = X_test_original[i].reshape(8, 8)
            real = y_test[i]
            pred = modelo_usar.predict(X_test_scaled[i:i+1])[0]
            proba = modelo_usar.predict_proba(X_test_scaled[i:i+1])[0]
            confianza = proba[pred] * 100
            
            # Mostrar imagen
            ax.imshow(imagen, cmap='gray')
            
            # Color del título según si es correcto o no
            if pred == real:
                color = 'green'
                correctas += 1
                simbolo = '✓'
            else:
                color = 'red'
                incorrectas += 1
                simbolo = '✗'
            
            ax.set_title(f'{simbolo} Real:{real} Pred:{pred}\n{confianza:.0f}% confianza',
                        fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('4_4_predicciones_ejemplo.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Predicciones de ejemplo guardadas: '4_4_predicciones_ejemplo.png'")
        
        print(f"\n📊 Resultados de la muestra:")
        print(f"  • Correctas: {correctas}/20 ({correctas/20*100:.0f}%)")
        print(f"  • Incorrectas: {incorrectas}/20 ({incorrectas/20*100:.0f}%)")

def main():
    """
    Función principal del proyecto
    """
    print("\n" + "=" * 80)
    print("PROYECTO: RECONOCIMIENTO DE DÍGITOS CON SCIKIT-LEARN")
    print("=" * 80)
    
    # Crear instancia del sistema
    sistema = SistemaReconocimientoDigitos()
    
    # Presentar framework
    sistema.presentacion_framework()
    
    # Ejecutar pipeline completo
    digits = sistema.cargar_y_explorar_datos()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = sistema.preprocesamiento_datos(digits)
    
    modelo = sistema.disenar_modelo()
    sistema.entrenar_modelo(X_train, y_train)
    
    mejor_modelo = sistema.optimizar_hiperparametros(X_train, y_train)
    
    y_pred, accuracy = sistema.evaluar_modelo(X_test, y_test)
    
    sistema.guardar_modelo()
    sistema.demostrar_predicciones(X_test_orig, X_test, y_test)
    
    print("\n" + "=" * 80)
    print("✅ PROYECTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    
    print(f"\n🎯 RESULTADOS FINALES:")
    print(f"  • Accuracy final: {accuracy*100:.2f}%")
    print(f"  • Framework: Scikit-learn")
    print(f"  • Modelo: Multi-Layer Perceptron")
    print(f"  • Dataset: Handwritten Digits (1797 imágenes)")
    
    print("\n📁 Archivos generados:")
    print("  • 4_4_dataset_ejemplos.png - Ejemplos del dataset")
    print("  • 4_4_pca_visualizacion.png - Visualización PCA")
    print("  • 4_4_entrenamiento_curvas.png - Curvas de entrenamiento")
    print("  • 4_4_evaluacion_modelo.png - Métricas de evaluación")
    print("  • 4_4_predicciones_ejemplo.png - Ejemplos de predicciones")
    print("  • 4_4_modelo_digitos.pkl - Modelo guardado")
    print("  • 4_4_scaler.pkl - Escalador guardado")
    
    print("\n💡 VENTAJAS DE SCIKIT-LEARN DEMOSTRADAS:")
    print("  ✓ API simple y consistente")
    print("  ✓ Herramientas de validación integradas (GridSearchCV)")
    print("  ✓ Fácil serialización de modelos")
    print("  ✓ Excelente para prototipado rápido")
    print("  ✓ Integración perfecta con NumPy y Pandas")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("  • Experimentar con arquitecturas más profundas")
    print("  • Probar con dataset MNIST completo (70,000 imágenes)")
    print("  • Implementar data augmentation")
    print("  • Desplegar modelo en producción")
    print("  • Crear API REST para el modelo")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
