# toolbox_ML.py

import pandas as pd

def describe_df(df):
    """
    Genera una descripción del DataFrame con información sobre las columnas.

    Argumentos:
    df (pd.DataFrame): DataFrame que se desea describir.

    Retorna:
    pd.DataFrame: DataFrame con información sobre cada columna:
        - Nombre de la columna
        - Tipo de dato
        - Porcentaje de valores nulos
        - Cantidad de valores únicos
        - Porcentaje de cardinalidad
    """
    description = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Null Percentage (%)": [(df[col].isnull().sum() / len(df)) * 100 for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns],
        "Cardinality (%)": [(df[col].nunique() / len(df)) * 100 for col in df.columns],
    })
    return description

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las variables de un DataFrame en tipos sugeridos.

    Argumentos:
    df (pd.DataFrame): DataFrame cuyas variables se desean clasificar.
    umbral_categoria (int): Umbral para considerar una variable como categórica.
    umbral_continua (float): Umbral para considerar una variable como continua.

    Retorna:
    pd.DataFrame: DataFrame con columnas:
        - "nombre_variable": Nombre de las columnas originales.
        - "tipo_sugerido": Tipo sugerido para cada variable.
    """
    resultado = []

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = cardinalidad / len(df)

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numérica Continua"
        else:
            tipo = "Numérica Discreta"

        resultado.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultado)


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def get_features_cat_regression(dataframe, target_col, pvalue=0.05):
    """
    Identifica columnas categóricas (numéricas con baja cardinalidad o explícitamente categóricas)
    que tienen una relación significativa con una columna numérica objetivo usando la prueba de Chi-cuadrado.

    Parámetros:
    - dataframe: pd.DataFrame. El DataFrame de entrada.
    - target_col: str. Nombre de la columna objetivo, debe ser numérica.
    - pvalue: float. Nivel de significancia para la prueba Chi-cuadrado.

    Retorna:
    - Una lista de columnas categóricas significativas o None si no se encuentran.
    """
    # Dividir columnas explícitamente categóricas y numéricas
    explicit_categorical_cols = [col for col in dataframe.select_dtypes(include=['object', 'category']).columns if col != target_col]
    potential_categorical_cols = [col for col in dataframe.select_dtypes(include=[np.number]).columns
                                   if col != target_col and dataframe[col].nunique() <= 10]

    # Combinar ambas listas
    all_categorical_cols = explicit_categorical_cols + potential_categorical_cols

    if not all_categorical_cols:
        print("Error: No se encontraron columnas categóricas o numéricas con baja cardinalidad.")
        return None

    significant_features = []

    for col in all_categorical_cols:
        try:
            # Crear una tabla de contingencia
            contingency_table = pd.crosstab(dataframe[col], pd.cut(dataframe[target_col], bins=5))

            # Realizar la prueba de Chi-cuadrado
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Comprobar si el p-valor es menor que el umbral dado
            if p < pvalue:
                significant_features.append(col)

        except Exception as e:
            print(f"Advertencia: Error al procesar la columna '{col}': {e}")

    if not significant_features:
        print("No se encontraron columnas categóricas con una relación significativa con el target.")
        return []

    return significant_features