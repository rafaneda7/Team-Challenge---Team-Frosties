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
