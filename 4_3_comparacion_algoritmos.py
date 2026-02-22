"""
4.3 Algoritmos y Técnicas de Machine Learning
Análisis Comparativo de Algoritmos
- Regresión Logística (Clasificación)
- SVM - Support Vector Machine (Clasificación)
- K-Means (Clustering)

Datasets utilizados:
- Iris Dataset (clasificación)
- Wine Dataset (clasificación y clustering)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, silhouette_score, davies_bouldin_score)
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

class ComparadorAlgoritmos:
    """
    Clase para comparar diferentes algoritmos de Machine Learning
    """
    
    def __init__(self):
        self.resultados = {
            'algoritmo': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'tiempo_entrenamiento': [],
            'tiempo_prediccion': []
        }
    
    def cargar_datasets(self):
        """
        Cargar los datasets para el análisis
        """
        print("=" * 80)
        print("CARGANDO DATASETS")
        print("=" * 80)
        
        # Dataset 1: Iris (clasificación)
        iris = load_iris()
        df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_iris['species'] = iris.target
        df_iris['species_name'] = df_iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("\n📊 Dataset 1: Iris (Clasificación de especies de flores)")
        print(f"   • Muestras: {len(df_iris)}")
        print(f"   • Características: {len(iris.feature_names)}")
        print(f"   • Clases: {len(iris.target_names)} - {iris.target_names}")
        print(f"   • Descripción: {iris.DESCR.split(chr(10))[0]}")
        
        # Dataset 2: Wine (clasificación y clustering)
        wine = load_wine()
        df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
        df_wine['class'] = wine.target
        df_wine['class_name'] = df_wine['class'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
        
        print("\n🍷 Dataset 2: Wine (Clasificación de vinos)")
        print(f"   • Muestras: {len(df_wine)}")
        print(f"   • Características: {len(wine.feature_names)}")
        print(f"   • Clases: {len(wine.target_names)} - {wine.target_names}")
        
        return df_iris, iris, df_wine, wine
    
    def visualizar_datasets(self, df_iris, df_wine):
        """
        Visualizar los datasets
        """
        print("\n" + "=" * 80)
        print("VISUALIZACIÓN DE DATASETS")
        print("=" * 80)
        
        fig = plt.figure(figsize=(16, 10))
        
        # Visualización Iris
        ax1 = plt.subplot(2, 3, 1)
        for species in df_iris['species_name'].unique():
            subset = df_iris[df_iris['species_name'] == species]
            plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], 
                       label=species, alpha=0.6, s=50)
        plt.xlabel('Longitud del Sépalo (cm)')
        plt.ylabel('Ancho del Sépalo (cm)')
        plt.title('Iris: Sépalo', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        for species in df_iris['species_name'].unique():
            subset = df_iris[df_iris['species_name'] == species]
            plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], 
                       label=species, alpha=0.6, s=50)
        plt.xlabel('Longitud del Pétalo (cm)')
        plt.ylabel('Ancho del Pétalo (cm)')
        plt.title('Iris: Pétalo', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        df_iris['species_name'].value_counts().plot(kind='bar', color=['#e74c3c', '#3498db', '#2ecc71'])
        plt.title('Iris: Distribución de Clases', fontweight='bold')
        plt.xlabel('Especies')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        
        # Visualización Wine con PCA
        pca = PCA(n_components=2)
        wine_pca = pca.fit_transform(df_wine.drop(['class', 'class_name'], axis=1))
        
        ax4 = plt.subplot(2, 3, 4)
        for wine_class in df_wine['class'].unique():
            mask = df_wine['class'] == wine_class
            plt.scatter(wine_pca[mask, 0], wine_pca[mask, 1], 
                       label=f'Clase {wine_class}', alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Wine: PCA 2D', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        df_wine['class_name'].value_counts().plot(kind='bar', color=['#9b59b6', '#f39c12', '#1abc9c'])
        plt.title('Wine: Distribución de Clases', fontweight='bold')
        plt.xlabel('Clases')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=0)
        
        ax6 = plt.subplot(2, 3, 6)
        correlacion_wine = df_wine.drop(['class', 'class_name'], axis=1).corr()
        sns.heatmap(correlacion_wine, cmap='coolwarm', center=0, cbar_kws={'shrink': 0.8})
        plt.title('Wine: Correlación (top features)', fontweight='bold')
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig('4_3_datasets_visualizacion.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualización guardada: '4_3_datasets_visualizacion.png'")
    
    def preparar_datos(self, X, y):
        """
        Preparar datos para entrenamiento
        """
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def algoritmo_1_regresion_logistica(self, X_train, X_test, y_train, y_test, nombre_dataset):
        """
        ALGORITMO 1: REGRESIÓN LOGÍSTICA
        """
        print("\n" + "=" * 80)
        print("ALGORITMO 1: REGRESIÓN LOGÍSTICA")
        print("=" * 80)
        
        print("\n📖 Descripción:")
        print("   La Regresión Logística es un algoritmo de clasificación que utiliza")
        print("   la función logística (sigmoide) para predecir la probabilidad de que")
        print("   una muestra pertenezca a una clase determinada.")
        
        print("\n⚙️  Características:")
        print("   • Tipo: Clasificación supervisada")
        print("   • Complejidad: O(n × m) donde n=muestras, m=características")
        print("   • Ventajas: Simple, rápido, interpretable")
        print("   • Desventajas: Asume linealidad en el espacio de características")
        
        print(f"\n🔬 Entrenando en dataset: {nombre_dataset}")
        
        # Entrenar modelo
        inicio_train = time.time()
        modelo = LogisticRegression(max_iter=1000, random_state=42)
        modelo.fit(X_train, y_train)
        tiempo_train = time.time() - inicio_train
        
        # Predicciones
        inicio_pred = time.time()
        y_pred = modelo.predict(X_test)
        tiempo_pred = time.time() - inicio_pred
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Validación cruzada
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
        
        print(f"\n📊 Resultados:")
        print(f"   • Accuracy:               {accuracy:.4f}")
        print(f"   • Precision (weighted):   {precision:.4f}")
        print(f"   • Recall (weighted):      {recall:.4f}")
        print(f"   • F1-Score (weighted):    {f1:.4f}")
        print(f"   • CV Score (mean):        {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"   • Tiempo entrenamiento:   {tiempo_train*1000:.2f} ms")
        print(f"   • Tiempo predicción:      {tiempo_pred*1000:.2f} ms")
        
        # Guardar resultados
        self.resultados['algoritmo'].append('Regresión Logística')
        self.resultados['accuracy'].append(accuracy)
        self.resultados['precision'].append(precision)
        self.resultados['recall'].append(recall)
        self.resultados['f1_score'].append(f1)
        self.resultados['tiempo_entrenamiento'].append(tiempo_train)
        self.resultados['tiempo_prediccion'].append(tiempo_pred)
        
        return modelo, y_pred
    
    def algoritmo_2_svm(self, X_train, X_test, y_train, y_test, nombre_dataset):
        """
        ALGORITMO 2: SUPPORT VECTOR MACHINE (SVM)
        """
        print("\n" + "=" * 80)
        print("ALGORITMO 2: SUPPORT VECTOR MACHINE (SVM)")
        print("=" * 80)
        
        print("\n📖 Descripción:")
        print("   SVM busca el hiperplano óptimo que maximiza el margen entre clases.")
        print("   Utiliza vectores de soporte (muestras más cercanas al límite de decisión)")
        print("   para definir la frontera entre clases.")
        
        print("\n⚙️  Características:")
        print("   • Tipo: Clasificación supervisada")
        print("   • Complejidad: O(n² × m) a O(n³ × m)")
        print("   • Ventajas: Efectivo en espacios de alta dimensión, robusto")
        print("   • Desventajas: Computacionalmente costoso, sensible a escalado")
        print("   • Kernel: RBF (Radial Basis Function)")
        
        print(f"\n🔬 Entrenando en dataset: {nombre_dataset}")
        
        # Entrenar modelo
        inicio_train = time.time()
        modelo = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        modelo.fit(X_train, y_train)
        tiempo_train = time.time() - inicio_train
        
        # Predicciones
        inicio_pred = time.time()
        y_pred = modelo.predict(X_test)
        tiempo_pred = time.time() - inicio_pred
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Validación cruzada
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
        
        print(f"\n📊 Resultados:")
        print(f"   • Accuracy:               {accuracy:.4f}")
        print(f"   • Precision (weighted):   {precision:.4f}")
        print(f"   • Recall (weighted):      {recall:.4f}")
        print(f"   • F1-Score (weighted):    {f1:.4f}")
        print(f"   • CV Score (mean):        {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"   • Tiempo entrenamiento:   {tiempo_train*1000:.2f} ms")
        print(f"   • Tiempo predicción:      {tiempo_pred*1000:.2f} ms")
        print(f"   • Vectores de soporte:    {len(modelo.support_)}")
        
        # Guardar resultados
        self.resultados['algoritmo'].append('SVM (RBF)')
        self.resultados['accuracy'].append(accuracy)
        self.resultados['precision'].append(precision)
        self.resultados['recall'].append(recall)
        self.resultados['f1_score'].append(f1)
        self.resultados['tiempo_entrenamiento'].append(tiempo_train)
        self.resultados['tiempo_prediccion'].append(tiempo_pred)
        
        return modelo, y_pred
    
    def algoritmo_3_kmeans(self, X, y_true, nombre_dataset):
        """
        ALGORITMO 3: K-MEANS (CLUSTERING)
        """
        print("\n" + "=" * 80)
        print("ALGORITMO 3: K-MEANS CLUSTERING")
        print("=" * 80)
        
        print("\n📖 Descripción:")
        print("   K-Means es un algoritmo de clustering no supervisado que agrupa")
        print("   muestras en K clusters basándose en la similitud de características.")
        print("   Minimiza la varianza intra-cluster.")
        
        print("\n⚙️  Características:")
        print("   • Tipo: Clustering no supervisado")
        print("   • Complejidad: O(n × k × i × m) donde i=iteraciones")
        print("   • Ventajas: Simple, escalable, rápido")
        print("   • Desventajas: Sensible a inicialización, requiere especificar K")
        
        print(f"\n🔬 Entrenando en dataset: {nombre_dataset}")
        
        n_clusters = len(np.unique(y_true))
        print(f"   • Número de clusters: {n_clusters}")
        
        # Entrenar modelo
        inicio_train = time.time()
        modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = modelo.fit_predict(X)
        tiempo_train = time.time() - inicio_train
        
        # Métricas para clustering
        silhouette = silhouette_score(X, y_pred)
        davies_bouldin = davies_bouldin_score(X, y_pred)
        inertia = modelo.inertia_
        
        # Intentar mapear clusters a clases reales para calcular accuracy
        from scipy.stats import mode
        labels_map = {}
        for cluster_id in range(n_clusters):
            mask = (y_pred == cluster_id)
            if mask.sum() > 0:
                labels_map[cluster_id] = mode(y_true[mask], keepdims=True)[0][0]
        
        y_pred_mapped = np.array([labels_map.get(cluster, cluster) for cluster in y_pred])
        accuracy = accuracy_score(y_true, y_pred_mapped)
        
        print(f"\n📊 Resultados:")
        print(f"   • Accuracy (mapeado):     {accuracy:.4f}")
        print(f"   • Silhouette Score:       {silhouette:.4f} (rango: [-1, 1], mejor: 1)")
        print(f"   • Davies-Bouldin Index:   {davies_bouldin:.4f} (mejor: más bajo)")
        print(f"   • Inertia:                {inertia:.2f}")
        print(f"   • Iteraciones:            {modelo.n_iter_}")
        print(f"   • Tiempo entrenamiento:   {tiempo_train*1000:.2f} ms")
        
        print(f"\n💡 Interpretación:")
        if silhouette > 0.7:
            print("   • Excelente separación entre clusters")
        elif silhouette > 0.5:
            print("   • Buena separación entre clusters")
        elif silhouette > 0.3:
            print("   • Separación moderada entre clusters")
        else:
            print("   • Clusters pueden sobrelaparse")
        
        return modelo, y_pred, silhouette
    
    def comparar_resultados(self):
        """
        Comparar resultados de todos los algoritmos
        """
        print("\n" + "=" * 80)
        print("COMPARACIÓN DE ALGORITMOS")
        print("=" * 80)
        
        df_resultados = pd.DataFrame(self.resultados)
        
        print("\n📊 TABLA COMPARATIVA:")
        print("-" * 80)
        print(df_resultados.to_string(index=False))
        print("-" * 80)
        
        # Visualización comparativa
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        colores = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Gráfico 1: Accuracy
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df_resultados['algoritmo'], df_resultados['accuracy'], color=colores)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Exactitud de Predicción', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=15)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 2: Precision
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df_resultados['algoritmo'], df_resultados['precision'], color=colores)
        ax2.set_ylabel('Precision')
        ax2.set_title('Precisión', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=15)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 3: Recall
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df_resultados['algoritmo'], df_resultados['recall'], color=colores)
        ax3.set_ylabel('Recall')
        ax3.set_title('Sensibilidad', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.tick_params(axis='x', rotation=15)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 4: F1-Score
        ax4 = axes[1, 0]
        bars4 = ax4.bar(df_resultados['algoritmo'], df_resultados['f1_score'], color=colores)
        ax4.set_ylabel('F1-Score')
        ax4.set_title('F1-Score', fontweight='bold')
        ax4.set_ylim([0, 1])
        ax4.tick_params(axis='x', rotation=15)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 5: Tiempo de entrenamiento
        ax5 = axes[1, 1]
        bars5 = ax5.bar(df_resultados['algoritmo'], 
                        df_resultados['tiempo_entrenamiento'] * 1000, color=colores)
        ax5.set_ylabel('Tiempo (ms)')
        ax5.set_title('Tiempo de Entrenamiento', fontweight='bold')
        ax5.tick_params(axis='x', rotation=15)
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 6: Radar chart comparativo
        ax6 = axes[1, 2]
        # Normalizar tiempo de entrenamiento para el radar chart
        max_tiempo = max(df_resultados['tiempo_entrenamiento'])
        tiempo_norm = [1 - (t / max_tiempo) for t in df_resultados['tiempo_entrenamiento']]
        
        categorias = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Velocidad']
        num_vars = len(categorias)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        for i, algo in enumerate(df_resultados['algoritmo']):
            valores = [
                df_resultados.iloc[i]['accuracy'],
                df_resultados.iloc[i]['precision'],
                df_resultados.iloc[i]['recall'],
                df_resultados.iloc[i]['f1_score'],
                tiempo_norm[i]
            ]
            valores += valores[:1]
            ax6.plot(angles, valores, 'o-', linewidth=2, label=algo, color=colores[i])
            ax6.fill(angles, valores, alpha=0.15, color=colores[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categorias, size=8)
        ax6.set_ylim(0, 1)
        ax6.set_title('Comparación Multidimensional', fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig('4_3_comparacion_algoritmos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualización guardada: '4_3_comparacion_algoritmos.png'")
        
        # Análisis y conclusiones
        print("\n" + "=" * 80)
        print("ANÁLISIS Y CONCLUSIONES")
        print("=" * 80)
        
        # Mejor algoritmo por métrica
        mejor_accuracy_idx = df_resultados['accuracy'].idxmax()
        mejor_precision_idx = df_resultados['precision'].idxmax()
        mejor_recall_idx = df_resultados['recall'].idxmax()
        mejor_f1_idx = df_resultados['f1_score'].idxmax()
        mas_rapido_idx = df_resultados['tiempo_entrenamiento'].idxmin()
        
        print(f"\n🏆 MEJORES ALGORITMOS POR MÉTRICA:")
        print(f"   • Mejor Accuracy:         {df_resultados.iloc[mejor_accuracy_idx]['algoritmo']} ({df_resultados.iloc[mejor_accuracy_idx]['accuracy']:.4f})")
        print(f"   • Mejor Precision:        {df_resultados.iloc[mejor_precision_idx]['algoritmo']} ({df_resultados.iloc[mejor_precision_idx]['precision']:.4f})")
        print(f"   • Mejor Recall:           {df_resultados.iloc[mejor_recall_idx]['algoritmo']} ({df_resultados.iloc[mejor_recall_idx]['recall']:.4f})")
        print(f"   • Mejor F1-Score:         {df_resultados.iloc[mejor_f1_idx]['algoritmo']} ({df_resultados.iloc[mejor_f1_idx]['f1_score']:.4f})")
        print(f"   • Más Rápido:             {df_resultados.iloc[mas_rapido_idx]['algoritmo']} ({df_resultados.iloc[mas_rapido_idx]['tiempo_entrenamiento']*1000:.2f} ms)")
        
        print(f"\n💡 RECOMENDACIONES:")
        print(f"   • Para máxima precisión: {df_resultados.iloc[mejor_accuracy_idx]['algoritmo']}")
        print(f"   • Para velocidad crítica: {df_resultados.iloc[mas_rapido_idx]['algoritmo']}")
        print(f"   • Para balance: {df_resultados.iloc[mejor_f1_idx]['algoritmo']}")
        
        print("\n📝 CONSIDERACIONES:")
        print("   • Regresión Logística: Excelente para líneas base, rápido, interpretable")
        print("   • SVM: Superior en espacios complejos, robusto pero costoso")
        print("   • K-Means: Útil para descubrimiento de patrones sin etiquetas")

def main():
    """
    Función principal
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPARATIVO DE ALGORITMOS DE MACHINE LEARNING")
    print("=" * 80)
    
    # Crear instancia del comparador
    comparador = ComparadorAlgoritmos()
    
    # Cargar datasets
    df_iris, iris, df_wine, wine = comparador.cargar_datasets()
    comparador.visualizar_datasets(df_iris, df_wine)
    
    # Preparar datos de Iris para clasificación
    X_iris = iris.data
    y_iris = iris.target
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = comparador.preparar_datos(X_iris, y_iris)
    
    print(f"\n✓ Datos preparados: {len(X_train_iris)} entrenamiento, {len(X_test_iris)} prueba")
    
    # Probar los tres algoritmos
    modelo_lr, pred_lr = comparador.algoritmo_1_regresion_logistica(
        X_train_iris, X_test_iris, y_train_iris, y_test_iris, "Iris"
    )
    
    modelo_svm, pred_svm = comparador.algoritmo_2_svm(
        X_train_iris, X_test_iris, y_train_iris, y_test_iris, "Iris"
    )
    
    # Preparar datos de Wine para clustering
    X_wine_scaled = StandardScaler().fit_transform(wine.data)
    modelo_kmeans, pred_kmeans, silhouette = comparador.algoritmo_3_kmeans(
        X_wine_scaled, wine.target, "Wine"
    )
    
    # Comparar resultados
    comparador.comparar_resultados()
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    print("\n📁 Archivos generados:")
    print("   • 4_3_datasets_visualizacion.png - Visualización de datasets")
    print("   • 4_3_comparacion_algoritmos.png - Comparación completa")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
