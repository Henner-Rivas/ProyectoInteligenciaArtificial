#!/usr/bin/env python3
"""
Script de prueba para verificar instalación de dependencias
"""

print("Verificando instalación de bibliotecas...")
print("=" * 60)

try:
    import numpy
    print("✓ NumPy", numpy.__version__)
except ImportError as e:
    print("✗ NumPy no disponible:", e)

try:
    import pandas
    print("✓ Pandas", pandas.__version__)
except ImportError as e:
    print("✗ Pandas no disponible:", e)

try:
    import sklearn
    print("✓ Scikit-learn", sklearn.__version__)
except ImportError as e:
    print("✗ Scikit-learn no disponible:", e)

try:
    import scipy
    print("✓ SciPy", scipy.__version__)
except ImportError as e:
    print("✗ SciPy no disponible:", e)

try:
    import matplotlib
    print("✓ Matplotlib", matplotlib.__version__)
except ImportError as e:
    print("✗ Matplotlib no disponible:", e)

try:
    import seaborn
    print("✓ Seaborn", seaborn.__version__)
except ImportError as e:
    print("✗ Seaborn no disponible:", e)

try:
    import joblib
    print("✓ Joblib", joblib.__version__)
except ImportError as e:
    print("✗ Joblib no disponible:", e)

print("=" * 60)
print("✓ Todas las bibliotecas necesarias están instaladas!")
print("\nPuedes ejecutar los scripts principales:")
print("  python3 4_1_fundamentos_ml.py")
print("  python3 4_2_aplicacion_salud.py")
print("  python3 4_3_comparacion_algoritmos.py")
print("  python3 4_4_framework_sklearn.py")
