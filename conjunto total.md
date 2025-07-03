Resumen súper completo de las operaciones básicas y fundamentales de Pandas. Estructurado por temas, como si fuera un flujo de trabajo de análisis de datos.

---

### **Resumen Intensivo de Pandas para tu Parcial de Fundamentos de Ciencia de Datos**

Este documento es tu guía de supervivencia. Cubre las operaciones más comunes que encontrarás en un análisis de datos básico, desde la carga hasta la exportación de los datos.

#### **0. Importaciones Esenciales**

Antes de empezar, asegúrate de importar las librerías que vas a usar. Es una buena práctica hacerlo al inicio de tu notebook.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm # Si lo necesitas para modelos estadísticos
```

---

### **1. Creación e Inspección Inicial del DataFrame**

Lo primero es cargar tus datos y darles un vistazo rápido para entender su estructura.

*   **Crear un DataFrame desde un diccionario (útil para pruebas rápidas):**
    ```python
    data = {'nombre': ['Ana', 'Luis', 'Pedro', 'Marta'],
            'edad': [28, 34, 29, 42],
            'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia']}
    df = pd.DataFrame(data)
    ```

*   **Leer datos desde un archivo:**
    ```python
    # Desde un CSV
    df_csv = pd.read_csv('ruta/a/tu/archivo.csv')
    # Parámetros útiles:
    # sep=';' si el separador es punto y coma.
    # header=None si no hay encabezado.
    # names=['col1', 'col2'] para nombrar las columnas si no hay encabezado.
    # index_col='nombre_columna' para usar una columna como índice.

    # Desde un Excel
    df_excel = pd.read_excel('ruta/a/tu/archivo.xlsx', sheet_name='NombreDeLaHoja')
    ```

*   **Inspección Rápida (los primeros comandos que siempre debes ejecutar):**
    ```python
    # Ver las primeras 5 filas
    print(df.head())

    # Ver las últimas 5 filas
    print(df.tail())

    # Ver las dimensiones (filas, columnas)
    print(df.shape)

    # Resumen conciso: tipos de datos, valores no nulos, uso de memoria (¡CRUCIAL!)
    print(df.info())

    # Resumen estadístico de las columnas numéricas (media, desviación, cuartiles, etc.)
    print(df.describe())

    # Ver los nombres de las columnas
    print(df.columns)

    # Ver los tipos de datos de cada columna
    print(df.dtypes)
    ```

---

### **2. Selección y Filtrado de Datos (Indexing & Slicing)**

Cómo acceder a las partes del DataFrame que te interesan.

*   **Seleccionar Columnas:**
    ```python
    # Seleccionar una columna (devuelve una Serie de Pandas)
    series_ciudad = df['ciudad']

    # Seleccionar múltiples columnas (devuelve un DataFrame)
    # ¡OJO con los dobles corchetes!
    sub_df = df[['nombre', 'ciudad']]
    ```

*   **Seleccionar Filas (usando `loc` e `iloc`):**
    *   `.loc`: Selección por **etiqueta/nombre** del índice.
    *   `.iloc`: Selección por **posición** numérica (entera).

    ```python
    # Seleccionar la primera fila por su posición (índice 0)
    fila_0 = df.iloc[0]

    # Seleccionar las primeras 3 filas
    primeras_3_filas = df.iloc[0:3]

    # Si el índice fuera el nombre de la persona:
    # df.set_index('nombre', inplace=True)
    # fila_ana = df.loc['Ana']

    # Seleccionar una celda específica: df.loc[etiqueta_fila, etiqueta_columna]
    edad_de_pedro = df.loc[2, 'edad'] # Fila con índice 2, columna 'edad'

    # Seleccionar una celda específica por posición: df.iloc[pos_fila, pos_columna]
    ciudad_de_luis = df.iloc[1, 2] # Fila 1, columna 2
    ```

*   **Filtrado por Condiciones (Boolean Indexing):**
    ```python
    # Personas mayores de 30 años
    mayores_30 = df[df['edad'] > 30]

    # Personas de Madrid
    de_madrid = df[df['ciudad'] == 'Madrid']

    # Combinar condiciones: & (Y), | (O)
    # ¡OJO con los paréntesis en cada condición!
    # Personas de Madrid que son mayores de 28 años
    madrid_y_mayores = df[(df['ciudad'] == 'Madrid') & (df['edad'] > 28)]

    # Personas que son de Madrid O de Valencia
    madrid_o_valencia = df[df['ciudad'].isin(['Madrid', 'Valencia'])]
    ```

---

### **3. Limpieza y Transformación de Datos**

Esta es la parte más importante y donde más tiempo se invierte.

*   **Manejo de Valores Faltantes (`NaN` - Not a Number):**
    ```python
    # 1. Detección: Contar cuántos NaN hay por columna
    print(df.isnull().sum())
    # O también df.isna().sum()

    # 2. Eliminación:
    # Eliminar cualquier fila que contenga al menos un NaN
    df_sin_nan_filas = df.dropna()
    # Eliminar cualquier columna que contenga al menos un NaN
    df_sin_nan_cols = df.dropna(axis=1)

    # 3. Imputación (Relleno):
    # Rellenar todos los NaN con un valor específico (ej. 0)
    df_relleno_0 = df.fillna(0)
    # Rellenar los NaN de una columna con la media de esa columna
    media_edad = df['edad'].mean()
    df['edad'].fillna(media_edad, inplace=True) # inplace=True modifica el df original
    # Rellenar con el valor anterior (forward fill)
    df.fillna(method='ffill', inplace=True)
    ```

*   **Manejo de Duplicados:**
    ```python
    # Detección: Contar cuántas filas están duplicadas
    print(df.duplicated().sum())

    # Eliminación: Eliminar filas duplicadas, manteniendo la primera aparición
    df_sin_duplicados = df.drop_duplicates(keep='first')
    ```

*   **Cambio de Nombres de Columnas:**
    ```python
    # Renombrar columnas específicas
    df.rename(columns={'nombre': 'Nombre Completo', 'ciudad': 'Ciudad Residencia'}, inplace=True)

    # Renombrar todas las columnas (¡cuidado, la lista debe tener la misma longitud!)
    df.columns = ['Name', 'Age', 'City']
    ```

*   **Conversión de Tipos de Datos (`dtypes`):**
    ```python
    # Convertir una columna a tipo numérico (entero)
    df['edad'] = df['edad'].astype(int)

    # Convertir una columna de texto que representa fechas a tipo datetime
    df['fecha_registro'] = pd.to_datetime(df['fecha_registro'])
    # Ahora puedes acceder a propiedades de fecha: df['fecha_registro'].dt.year, .dt.month, etc.

    # Convertir a tipo 'category' (eficiente para columnas con pocos valores únicos)
    df['ciudad'] = df['ciudad'].astype('category')
    ```

*   **Modificar Datos y Crear Nuevas Columnas:**
    ```python
    # Crear una nueva columna basada en otras
    df['edad_en_5_anios'] = df['edad'] + 5

    # Aplicar una función a una columna con .apply() y una función lambda
    # Por ejemplo, categorizar la edad
    df['grupo_edad'] = df['edad'].apply(lambda x: 'Joven' if x < 30 else 'Adulto')

    # Modificar valores específicos usando .loc
    # Cambiar 'Madrid' por 'MAD' en la columna 'ciudad'
    df.loc[df['ciudad'] == 'Madrid', 'ciudad'] = 'MAD'

    # Reemplazar valores usando .map() con un diccionario (muy útil para codificar)
    df['ciudad_codificada'] = df['ciudad'].map({'MAD': 1, 'Barcelona': 2, 'Valencia': 3})
    ```

*   **Eliminar Columnas o Filas:**
    ```python
    # Eliminar una columna
    df.drop('columna_a_eliminar', axis=1, inplace=True) # axis=1 es para columnas

    # Eliminar una fila por su índice
    df.drop(3, axis=0, inplace=True) # axis=0 es para filas
    ```

---

### **4. Agrupación y Agregación (`groupby`)**

El patrón "Split-Apply-Combine" es fundamental para resumir datos.

*   **Agrupar por una columna y aplicar una agregación:**
    ```python
    # Calcular la edad media por ciudad
    edad_media_por_ciudad = df.groupby('ciudad')['edad'].mean()
    print(edad_media_por_ciudad)

    # Contar cuántas personas hay por ciudad
    conteo_por_ciudad = df.groupby('ciudad')['nombre'].count()
    # o de forma más simple:
    conteo_por_ciudad_v2 = df.groupby('ciudad').size()
    ```

*   **Agrupar por múltiples columnas:**
    ```python
    # Agrupar por ciudad y grupo_edad y calcular la suma de alguna otra variable 'ventas'
    # ventas_por_grupo = df.groupby(['ciudad', 'grupo_edad'])['ventas'].sum()
    ```

*   **Aplicar múltiples agregaciones a la vez con `.agg()`:**
    ```python
    # Para cada ciudad, calcular la edad media y la edad máxima
    agregaciones = df.groupby('ciudad')['edad'].agg(['mean', 'max', 'count'])
    print(agregaciones)
    ```

---

### **5. Combinación de DataFrames (`merge` y `concat`)**

*   **Concatenar (`concat`):** "Pegar" DataFrames uno debajo del otro (`axis=0`) o uno al lado del otro (`axis=1`).
    ```python
    df1 = ...
    df2 = ...
    # Apilar df2 debajo de df1 (deben tener columnas similares)
    df_completo = pd.concat([df1, df2], ignore_index=True) # ignore_index=True resetea el índice
    ```

*   **Unir (`merge`):** Similar a los JOIN de SQL.
    ```python
    df_clientes = pd.DataFrame({'id': [1, 2, 3], 'nombre': ['Ana', 'Luis', 'Pedro']})
    df_pedidos = pd.DataFrame({'id_cliente': [1, 1, 3], 'producto': ['Libro', 'Lápiz', 'Goma']})

    # Unir ambos DataFrames usando la columna clave
    df_merged = pd.merge(
        left=df_clientes,
        right=df_pedidos,
        left_on='id', # Columna clave en el df izquierdo
        right_on='id_cliente', # Columna clave en el df derecho
        how='inner' # Tipo de join: 'inner', 'outer', 'left', 'right'
    )
    ```

---

### **6. Operaciones Útiles Adicionales**

*   **Contar valores únicos en una columna (¡muy útil para variables categóricas!):**
    ```python
    print(df['ciudad'].value_counts())
    ```

*   **Obtener los valores únicos de una columna:**
    ```python
    print(df['ciudad'].unique())
    ```
*   **Contar el número de valores únicos:**
    ```python
    print(df['ciudad'].nunique())
    ```

*   **Ordenar el DataFrame por los valores de una columna:**
    ```python
    # Ordenar por edad, de mayor a menor
    df_ordenado = df.sort_values(by='edad', ascending=False)
    ```

*   **Resetear el índice:** A menudo, después de filtrar o agrupar, el índice queda desordenado.
    ```python
    df_filtrado = df[df['edad'] > 30]
    df_filtrado.reset_index(drop=True, inplace=True) # drop=True evita que el viejo índice se convierta en una columna
    ```
---

### **7. Visualización y Exportación**

*   **Visualización rápida con Pandas + Matplotlib:**
    ```python
    # Histograma de una columna
    df['edad'].hist(bins=10)
    plt.show()

    # Gráfico de barras de conteos
    df['ciudad'].value_counts().plot(kind='bar')
    plt.show()
    ```

*   **Visualización con Seaborn (más potente y estético):**
    ```python
    # Histograma
    sns.histplot(data=df, x='edad', kde=True) # kde=True añade una curva de densidad
    plt.show()

    # Gráfico de cajas (boxplot) para ver la distribución por categoría
    sns.boxplot(data=df, x='ciudad', y='edad')
    plt.show()

    # Gráfico de dispersión (scatterplot) para ver la relación entre dos variables
    sns.scatterplot(data=df, x='edad', y='otra_variable_numerica')
    plt.show()
    ```

*   **Guardar el DataFrame limpio:**
    ```python
    # Guardar a CSV
    df.to_csv('datos_limpios.csv', index=False) # index=False para no guardar el índice como una columna

    # Guardar a Excel
    df.to_excel('datos_limpios.xlsx', index=False)
    ```

---



# Guía de operaciones básicas en pandas para Ciencia de Datos

Esta guía está pensada para usarse como referencia rápida. Contiene operaciones comunes de manipulación de datos en `pandas`, con ejemplos sintéticos para facilitar su búsqueda y comprensión. Está organizada por categoría.

---

## 1. Carga y guardado de datos

```python
import pandas as pd

df = pd.read_csv("archivo.csv")
df.to_csv("salida.csv", index=False)
```

## 2. Exploración básica

```python
df.head()         # Primeras 5 filas
df.tail()         # Últimas 5 filas
df.info()         # Estructura del DataFrame
df.describe()     # Estadísticas descriptivas
df.shape          # Dimensiones
df.columns        # Nombres de columnas
df.dtypes         # Tipos de datos
df["col"].value_counts()  # Frecuencia de valores
```

## 3. Selección y filtrado

```python
df["col"]                  # Columna por nombre
df[["col1", "col2"]]      # Varias columnas
df.loc[3]                 # Fila por etiqueta
df.iloc[3]                # Fila por índice

df[df["edad"] > 30]       # Filtro booleano
```

## 4. Modificación de datos

```python
df["col"] = df["col"].astype(float)    # Cambio de tipo
df.rename(columns={"old": "new"}, inplace=True) # Renombrar columnas

df.at[3, "col"] = 99       # Cambiar un valor específico
df.loc[df["sexo"] == "M", "sexo"] = "masculino"  # Reemplazo condicional
```

## 5. Valores faltantes

```python
df.isna().sum()           # Cantidad de NaNs por columna
df.dropna()               # Eliminar filas con NaNs
df.fillna(0)              # Reemplazar NaNs
```

## 6. Duplicados

```python
df.duplicated()           # Booleano por fila duplicada
df.drop_duplicates()      # Eliminar duplicados
```

## 7. Reemplazo y transformaciones

```python
df["col"].replace({"a": 1, "b": 2})
df["sexo"].map({"M": 0, "F": 1})
df["col"].apply(lambda x: x**2)  # Funciones personalizadas
```

## 8. Nuevas columnas

```python
df["col_suma"] = df["a"] + df["b"]
df["edad_mayor_30"] = df["edad"] > 30
```

## 9. Agrupamiento y agregación

```python
df.groupby("sexo")["ingreso"].mean()
df.groupby(["sexo", "region"]).agg({"ingreso": ["mean", "std"]})
```

## 10. Ordenamiento

```python
df.sort_values("ingreso", ascending=False)
df.sort_index()
```

## 11. Combinación de DataFrames

```python
pd.concat([df1, df2])                    # Apila por filas
df1.merge(df2, on="id")                 # Join por columna
df1.join(df2.set_index("id"), on="id")  # Join por índice
```

## 12. Pivot y reshape

```python
df.pivot(index="fecha", columns="producto", values="ventas")
df.melt(id_vars="id", var_name="variable", value_name="valor")
df.stack() / df.unstack()
```

## 13. Fechas y tiempos

```python
df["fecha"] = pd.to_datetime(df["fecha"])
df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
```

## 14. Codificación de variables

```python
# One-hot encoding
df = pd.get_dummies(df, columns=["categoria"])

# Dicotomización
import numpy as np
df["alto"] = np.where(df["altura"] > 1.75, 1, 0)

# Binning
df["edad_grupo"] = pd.cut(df["edad"], bins=[0,18,65,99], labels=["menor","adulto","mayor"])
```

## 15. Estadísticas simples

```python
df["ingreso"].mean()
df["ingreso"].median()
df["ingreso"].std()
df["ingreso"].quantile(0.75)
```

---

**Consejo:** usá `Ctrl + F` para buscar directamente por nombre de función o categoría durante el examen (ej. "fillna", "groupby", "pivot", etc).

Podés complementar esta guía con referencias oficiales:

- [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- [https://www.w3schools.com/python/pandas](https://www.w3schools.com/python/pandas)
- [https://www.datacamp.com](https://www.datacamp.com)

También podés revisar recursos como cheat sheets de pandas o notebooks de ejercicios prácticos.



# Guía de Referencia Rápida - Pandas para Ciencia de Datos

## 📋 Importaciones Básicas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
```

## 📁 1. CARGA Y GUARDADO DE DATOS

### Lectura de archivos
```python
# CSV
df = pd.read_csv('archivo.csv')
df = pd.read_csv('archivo.csv', sep=';', encoding='utf-8')
df = pd.read_csv('archivo.csv', index_col=0)  # Primera columna como índice
df = pd.read_csv('archivo.csv', usecols=['col1', 'col2'])  # Solo ciertas columnas
df = pd.read_csv('archivo.csv', nrows=1000)  # Solo primeras 1000 filas

# Excel
df = pd.read_excel('archivo.xlsx', sheet_name='Hoja1')
df = pd.read_excel('archivo.xlsx', sheet_name=0)  # Primera hoja

# JSON
df = pd.read_json('archivo.json')

# Desde URL
df = pd.read_csv('https://ejemplo.com/datos.csv')
```

### Guardado de archivos
```python
# CSV
df.to_csv('salida.csv', index=False)  # Sin índice
df.to_csv('salida.csv', sep=';', encoding='utf-8')

# Excel
df.to_excel('salida.xlsx', sheet_name='Datos', index=False)

# JSON
df.to_json('salida.json', orient='records')
```

## 🔍 2. EXPLORACIÓN Y VISUALIZACIÓN BÁSICA

### Información general del DataFrame
```python
# Primeras/últimas filas
df.head()      # Primeras 5 filas
df.head(10)    # Primeras 10 filas
df.tail()      # Últimas 5 filas
df.tail(3)     # Últimas 3 filas

# Información del DataFrame
df.info()          # Tipos de datos, memoria, valores no nulos
df.shape           # (filas, columnas)
df.columns         # Nombres de columnas
df.index           # Información del índice
df.dtypes          # Tipos de datos por columna
df.memory_usage()  # Uso de memoria por columna

# Estadísticas descriptivas
df.describe()                    # Solo columnas numéricas
df.describe(include='all')       # Todas las columnas
df.describe(include=['object'])  # Solo columnas de texto
df.describe(percentiles=[.1, .25, .5, .75, .9])  # Percentiles personalizados

# Valores únicos y conteos
df['columna'].value_counts()                    # Conteo de valores únicos
df['columna'].value_counts(normalize=True)      # Proporciones
df['columna'].value_counts(dropna=False)        # Incluir NaN
df['columna'].nunique()                         # Número de valores únicos
df['columna'].unique()                          # Array de valores únicos

# Información por columna
df['columna'].count()    # Valores no nulos
df['columna'].sum()      # Suma
df['columna'].mean()     # Media
df['columna'].median()   # Mediana
df['columna'].std()      # Desviación estándar
df['columna'].var()      # Varianza
df['columna'].min()      # Mínimo
df['columna'].max()      # Máximo
```

## 🎯 3. SELECCIÓN Y FILTRADO DE DATOS

### Selección de columnas
```python
# Una columna
df['columna']
df.columna  # Solo si el nombre no tiene espacios ni caracteres especiales

# Múltiples columnas
df[['col1', 'col2', 'col3']]

# Columnas por posición
df.iloc[:, 0]      # Primera columna
df.iloc[:, 0:3]    # Primeras 3 columnas
df.iloc[:, -1]     # Última columna
```

### Selección de filas
```python
# Por posición (iloc)
df.iloc[0]         # Primera fila
df.iloc[0:5]       # Primeras 5 filas
df.iloc[-1]        # Última fila
df.iloc[::2]       # Filas pares

# Por etiqueta (loc)
df.loc[0]          # Fila con índice 0
df.loc[0:4]        # Filas del índice 0 al 4 (inclusive)
df.loc['etiqueta'] # Fila con índice 'etiqueta'

# Combinando filas y columnas
df.iloc[0:5, 0:3]           # Primeras 5 filas, primeras 3 columnas
df.loc[0:4, 'col1':'col3']  # Filas 0-4, columnas col1 a col3
df.loc[:, 'col1':'col3']    # Todas las filas, columnas col1 a col3
```

### Filtrado con condiciones (Boolean Indexing)
```python
# Condiciones simples
df[df['edad'] > 25]
df[df['nombre'] == 'Juan']
df[df['salario'] >= 50000]

# Múltiples condiciones
df[(df['edad'] > 25) & (df['salario'] < 100000)]  # AND
df[(df['ciudad'] == 'Madrid') | (df['ciudad'] == 'Barcelona')]  # OR
df[~(df['edad'] < 18)]  # NOT (negación)

# Filtros con métodos de string
df[df['nombre'].str.contains('Ana')]
df[df['email'].str.endswith('.com')]
df[df['nombre'].str.startswith('M')]

# Filtros con isin()
ciudades = ['Madrid', 'Barcelona', 'Valencia']
df[df['ciudad'].isin(ciudades)]

# Filtros con between()
df[df['edad'].between(25, 65)]

# Filtros con query() - alternativa legible
df.query('edad > 25 and salario < 100000')
df.query('ciudad in ["Madrid", "Barcelona"]')
```

## ✏️ 4. MODIFICACIÓN DE COLUMNAS Y FILAS

### Asignar valores
```python
# Nueva columna
df['nueva_columna'] = 0
df['total'] = df['precio'] * df['cantidad']

# Modificar columna existente
df['columna'] = df['columna'] * 2

# Modificar valores específicos
df.loc[df['edad'] > 65, 'categoria'] = 'Senior'
df.loc[0, 'nombre'] = 'Nuevo Nombre'  # Fila específica

# Múltiples asignaciones
df.loc[df['salario'] < 30000, ['categoria', 'nivel']] = ['Junior', 1]
```

### Eliminar columnas y filas
```python
# Eliminar columnas
df.drop('columna', axis=1)  # No modifica el original
df.drop(['col1', 'col2'], axis=1, inplace=True)  # Modifica el original
del df['columna']  # Elimina permanentemente

# Eliminar filas
df.drop(0)  # Eliminar fila con índice 0
df.drop([0, 1, 2])  # Eliminar múltiples filas
df.drop(df[df['edad'] < 18].index)  # Eliminar filas que cumplen condición
```

## 🔄 5. CONVERSIÓN DE TIPOS DE DATOS

```python
# Convertir tipos
df['columna'] = df['columna'].astype('int64')
df['columna'] = df['columna'].astype('float64')
df['columna'] = df['columna'].astype('str')
df['columna'] = df['columna'].astype('category')

# Múltiples columnas a la vez
df = df.astype({'col1': 'int64', 'col2': 'float64', 'col3': 'str'})

# Conversiones numéricas con manejo de errores
df['columna'] = pd.to_numeric(df['columna'], errors='coerce')  # NaN si no se puede convertir
df['columna'] = pd.to_numeric(df['columna'], errors='ignore')  # Mantiene original si no se puede

# Fechas
df['fecha'] = pd.to_datetime(df['fecha'])
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
```

## 🏷️ 6. RENOMBRAR COLUMNAS E ÍNDICES

```python
# Renombrar columnas
df.rename(columns={'nombre_viejo': 'nombre_nuevo'})
df.rename(columns={'col1': 'nueva_col1', 'col2': 'nueva_col2'}, inplace=True)

# Renombrar todas las columnas
df.columns = ['nueva_col1', 'nueva_col2', 'nueva_col3']

# Modificar nombres de columnas
df.columns = df.columns.str.lower()  # Minúsculas
df.columns = df.columns.str.replace(' ', '_')  # Reemplazar espacios por _

# Renombrar índice
df.rename(index={0: 'primera_fila', 1: 'segunda_fila'})
df.index.name = 'ID'
```

## 🧹 7. LIMPIEZA Y PREPROCESAMIENTO

### Valores nulos (NaN)
```python
# Detectar valores nulos
df.isna()          # DataFrame booleano
df.isnull()        # Igual que isna()
df.isna().sum()    # Conteo de nulos por columna
df.isna().any()    # ¿Hay algún nulo por columna?
df.isna().all()    # ¿Son todos nulos por columna?

# Eliminar valores nulos
df.dropna()                    # Elimina filas con cualquier NaN
df.dropna(axis=1)              # Elimina columnas con cualquier NaN
df.dropna(subset=['col1'])     # Solo considera col1 para eliminar
df.dropna(how='all')           # Solo si toda la fila es NaN
df.dropna(thresh=3)            # Mantiene filas con al menos 3 valores no nulos

# Rellenar valores nulos
df.fillna(0)                   # Rellenar con 0
df.fillna({'col1': 0, 'col2': 'Desconocido'})  # Diferentes valores por columna
df.fillna(method='ffill')      # Forward fill (repetir último valor válido)
df.fillna(method='bfill')      # Backward fill (usar siguiente valor válido)
df['columna'].fillna(df['columna'].mean())  # Rellenar con la media
df['columna'].fillna(df['columna'].mode()[0])  # Rellenar con la moda
```

### Duplicados
```python
# Detectar duplicados
df.duplicated()                          # Series booleana
df.duplicated().sum()                    # Conteo de duplicados
df.duplicated(subset=['col1', 'col2'])   # Solo considerar ciertas columnas
df.duplicated(keep='first')              # Marcar duplicados excepto el primero
df.duplicated(keep='last')               # Marcar duplicados excepto el último
df.duplicated(keep=False)                # Marcar todos los duplicados

# Eliminar duplicados
df.drop_duplicates()                     # Eliminar filas duplicadas
df.drop_duplicates(subset=['col1'])      # Solo considerar col1
df.drop_duplicates(keep='last')          # Mantener último duplicado
```

### Reemplazo de valores
```python
# replace()
df.replace('valor_viejo', 'valor_nuevo')
df.replace(['val1', 'val2'], 'nuevo_valor')  # Múltiples a uno
df.replace({'val1': 'nuevo1', 'val2': 'nuevo2'})  # Diccionario de reemplazos
df['columna'].replace({'A': 1, 'B': 2, 'C': 3})  # En una columna específica

# map() - solo para Series
df['columna'] = df['columna'].map({'A': 1, 'B': 2, 'C': 3})

# Reemplazar con regex
df.replace(r'[^\w\s]', '', regex=True)  # Eliminar caracteres especiales
```

## ➕ 8. CREACIÓN DE COLUMNAS NUEVAS

```python
# Operaciones aritméticas
df['total'] = df['precio'] * df['cantidad']
df['descuento'] = df['precio'] * 0.1

# Operaciones condicionales
df['categoria_edad'] = df['edad'].apply(lambda x: 'Joven' if x < 30 else 'Adulto')

# Con np.where (más eficiente)
df['categoria'] = np.where(df['edad'] < 30, 'Joven', 'Adulto')

# Condiciones múltiples con np.select
condiciones = [
    df['edad'] < 18,
    (df['edad'] >= 18) & (df['edad'] < 65),
    df['edad'] >= 65
]
valores = ['Menor', 'Adulto', 'Senior']
df['grupo_edad'] = np.select(condiciones, valores, default='Desconocido')

# Combinando strings
df['nombre_completo'] = df['nombre'] + ' ' + df['apellido']

# Fechas y tiempo
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia_semana'] = df['fecha'].dt.day_name()
```

## 📊 9. AGRUPAMIENTO (GroupBy)

### Operaciones básicas
```python
# Agrupar por una columna
df.groupby('categoria').sum()      # Suma por grupo
df.groupby('categoria').mean()     # Media por grupo
df.groupby('categoria').count()    # Conteo por grupo
df.groupby('categoria').size()     # Tamaño de cada grupo

# Agrupar por múltiples columnas
df.groupby(['categoria', 'region']).sum()

# Operaciones específicas por columna
df.groupby('categoria')['ventas'].sum()
df.groupby('categoria')[['ventas', 'cantidad']].mean()
```

### Agregaciones avanzadas
```python
# agg() con múltiples funciones
df.groupby('categoria').agg({
    'ventas': ['sum', 'mean', 'count'],
    'cantidad': ['sum', 'max'],
    'precio': 'mean'
})

# Funciones personalizadas
df.groupby('categoria').agg({
    'ventas': lambda x: x.max() - x.min(),  # Rango
    'precio': ['mean', np.std]
})

# Renombrar columnas en agg
df.groupby('categoria').agg({
    'ventas': [('total_ventas', 'sum'), ('promedio_ventas', 'mean')],
    'cantidad': [('max_cantidad', 'max')]
})

# transform() - mantiene el tamaño original
df['ventas_promedio_categoria'] = df.groupby('categoria')['ventas'].transform('mean')

# apply() - máxima flexibilidad
def estadisticas_grupo(grupo):
    return pd.Series({
        'total': grupo['ventas'].sum(),
        'promedio': grupo['ventas'].mean(),
        'max': grupo['ventas'].max()
    })

df.groupby('categoria').apply(estadisticas_grupo)
```

## 🔗 10. COMBINACIÓN Y FUSIÓN DE DATAFRAMES

### merge() - SQL-style joins
```python
# Inner join (intersección)
df_resultado = pd.merge(df1, df2, on='columna_comun')
df_resultado = pd.merge(df1, df2, on=['col1', 'col2'])  # Múltiples columnas

# Diferentes tipos de join
pd.merge(df1, df2, on='id', how='inner')    # Solo coincidencias
pd.merge(df1, df2, on='id', how='left')     # Todas las filas de df1
pd.merge(df1, df2, on='id', how='right')    # Todas las filas de df2
pd.merge(df1, df2, on='id', how='outer')    # Todas las filas de ambos

# Columnas con nombres diferentes
pd.merge(df1, df2, left_on='id1', right_on='id2')

# Merge por índice
pd.merge(df1, df2, left_index=True, right_index=True)
```

### concat() - Concatenación
```python
# Concatenar verticalmente (apilar filas)
df_resultado = pd.concat([df1, df2])
df_resultado = pd.concat([df1, df2], ignore_index=True)  # Resetear índice

# Concatenar horizontalmente (añadir columnas)
df_resultado = pd.concat([df1, df2], axis=1)

# Con etiquetas
df_resultado = pd.concat([df1, df2], keys=['grupo1', 'grupo2'])
```

### join() - Join por índice
```python
# Join por índice (más simple que merge)
df_resultado = df1.join(df2)
df_resultado = df1.join(df2, how='left')
df_resultado = df1.join(df2, rsuffix='_right')  # Sufijo para columnas duplicadas
```

## 📈 11. ORDENAMIENTO

```python
# Ordenar por valores
df.sort_values('columna')                    # Ascendente
df.sort_values('columna', ascending=False)   # Descendente
df.sort_values(['col1', 'col2'])             # Múltiples columnas
df.sort_values(['col1', 'col2'], ascending=[True, False])  # Mixto

# Ordenar por índice
df.sort_index()                    # Ascendente
df.sort_index(ascending=False)     # Descendente

# Modificar el DataFrame original
df.sort_values('columna', inplace=True)
```

## 🔧 12. APLICACIÓN DE FUNCIONES

### apply()
```python
# En Series (columna)
df['nueva_col'] = df['columna'].apply(lambda x: x * 2)
df['nueva_col'] = df['columna'].apply(str.upper)  # Función incorporada

# En DataFrame
df.apply(lambda x: x.sum())  # Por columnas (axis=0, default)
df.apply(lambda x: x.sum(), axis=1)  # Por filas

# Función personalizada
def procesar_fila(fila):
    return fila['col1'] + fila['col2']

df['nueva_col'] = df.apply(procesar_fila, axis=1)

# applymap() - elemento por elemento
df.applymap(lambda x: str(x).upper())  # Todo el DataFrame
```

### map()
```python
# Solo para Series, mapeo de valores
df['columna_mapeada'] = df['columna'].map({'A': 1, 'B': 2, 'C': 3})

# Con función
df['columna_procesada'] = df['columna'].map(lambda x: x.upper())
```

## 🔄 13. RESHAPE DE DATOS

### pivot() y pivot_table()
```python
# pivot() - Datos ya agregados
df_pivot = df.pivot(index='fila', columns='columna', values='valores')

# pivot_table() - Con agregación automática
df_pivot = df.pivot_table(
    index='categoria',
    columns='mes',
    values='ventas',
    aggfunc='sum',
    fill_value=0
)

# Múltiples valores
df_pivot = df.pivot_table(
    index='categoria',
    columns='mes',
    values=['ventas', 'cantidad'],
    aggfunc={'ventas': 'sum', 'cantidad': 'mean'}
)
```

### melt() - Wide to Long
```python
# Convertir columnas en filas
df_long = df.melt(
    id_vars=['id', 'nombre'],      # Columnas que se mantienen
    value_vars=['ene', 'feb', 'mar'],  # Columnas a convertir
    var_name='mes',                # Nombre para la nueva columna de variables
    value_name='ventas'            # Nombre para la nueva columna de valores
)

# Melt simple
df_long = df.melt(id_vars=['id'], var_name='variable', value_name='valor')
```

### stack() y unstack()
```python
# stack() - Columnas a filas (MultiIndex)
df_stacked = df.stack()

# unstack() - Filas a columnas
df_unstacked = df.unstack()
df_unstacked = df.unstack(level=0)  # Nivel específico del índice
```

## 📅 14. TRABAJO CON FECHAS Y TIEMPOS

### Conversión a datetime
```python
# Conversión básica
df['fecha'] = pd.to_datetime(df['fecha'])
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

# Manejo de errores
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# Desde componentes separados
df['fecha'] = pd.to_datetime(df[['año', 'mes', 'dia']])
```

### Operaciones con fechas (dt accessor)
```python
# Extraer componentes
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek  # 0=Lunes
df['nombre_dia'] = df['fecha'].dt.day_name()
df['nombre_mes'] = df['fecha'].dt.month_name()
df['trimestre'] = df['fecha'].dt.quarter

# Operaciones
df['edad_dias'] = (pd.Timestamp.now() - df['fecha_nacimiento']).dt.days
df['semana_año'] = df['fecha'].dt.isocalendar().week

# Filtros por fecha
df[df['fecha'] > '2023-01-01']
df[df['fecha'].dt.year == 2023]
df[df['fecha'].dt.month.isin([6, 7, 8])]  # Verano
```

### Resample (para series temporales)
```python
# Agrupar por período temporal (requiere índice datetime)
df.set_index('fecha').resample('M').sum()    # Por mes
df.set_index('fecha').resample('Q').mean()   # Por trimestre
df.set_index('fecha').resample('Y').count()  # Por año
df.set_index('fecha').resample('D').ffill()  # Por día, rellenar
```

## 🔢 15. DICOTOMIZACIÓN (VARIABLES BINARIAS)

```python
# Crear variable binaria con condición
df['es_mayor'] = (df['edad'] >= 18).astype(int)
df['salario_alto'] = np.where(df['salario'] > 50000, 1, 0)

# Múltiples categorías a binarias
df['categoria_A'] = (df['categoria'] == 'A').astype(int)
df['categoria_B'] = (df['categoria'] == 'B').astype(int)

# Con map()
df['genero_num'] = df['genero'].map({'M': 1, 'F': 0})
```

## 🎯 16. ONE HOT ENCODING

```python
# get_dummies() - método más común
dummies = pd.get_dummies(df['categoria'])
df = pd.concat([df, dummies], axis=1)

# Con prefijo
dummies = pd.get_dummies(df['categoria'], prefix='cat')

# Múltiples columnas
dummies = pd.get_dummies(df[['categoria', 'region']])

# Eliminar una columna dummy (evitar multicolinealidad)
dummies = pd.get_dummies(df['categoria'], drop_first=True)

# Integrado en el DataFrame
df_encoded = pd.get_dummies(df, columns=['categoria', 'region'])

# Mantener datos originales
df_encoded = pd.get_dummies(df, columns=['categoria'], dummy_na=True)  # Incluir NaN
```

## 📦 17. BINNING (DISCRETIZACIÓN)

```python
# cut() - Intervalos de igual ancho
df['grupo_edad'] = pd.cut(df['edad'], bins=3, labels=['Joven', 'Adulto', 'Mayor'])
df['grupo_edad'] = pd.cut(df['edad'], bins=[0, 18, 65, 100])

# Con etiquetas personalizadas
df['grupo_salario'] = pd.cut(
    df['salario'], 
    bins=[0, 30000, 60000, 100000, float('inf')],
    labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
)

# qcut() - Cuantiles (igual frecuencia)
df['cuartil_salario'] = pd.qcut(df['salario'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
df['decil'] = pd.qcut(df['ventas'], q=10)

# Personalizado con apply
def categorizar_edad(edad):
    if edad < 18:
        return 'Menor'
    elif edad < 65:
        return 'Adulto'
    else:
        return 'Senior'

df['categoria_edad'] = df['edad'].apply(categorizar_edad)
```

## 📊 18. GRÁFICOS CON SEABORN (BÁSICOS)

### Configuración inicial
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Estilo
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.figure(figsize=(10, 6))
```

### Gráficos univariados
```python
# Histograma
sns.histplot(df['columna'])
sns.histplot(df['columna'], bins=20, kde=True)

# Boxplot
sns.boxplot(y=df['columna'])
sns.boxplot(x='categoria', y='valor', data=df)

# Violin plot
sns.violinplot(x='categoria', y='valor', data=df)

# Distribución
sns.distplot(df['columna'])  # Deprecated, usar histplot
sns.kdeplot(df['columna'])
```

### Gráficos bivariados
```python
# Scatterplot
sns.scatterplot(x='var1', y='var2', data=df)
sns.scatterplot(x='var1', y='var2', hue='categoria', data=df)

# Regplot (con línea de regresión)
sns.regplot(x='var1', y='var2', data=df)

# Lineplot
sns.lineplot(x='fecha', y='valor', data=df)
sns.lineplot(x='fecha', y='valor', hue='categoria', data=df)

# Barplot
sns.barplot(x='categoria', y='valor', data=df)
sns.countplot(x='categoria', data=df)
```

### Gráficos multivariados
```python
# Pairplot
sns.pairplot(df)
sns.pairplot(df, hue='categoria')

# Heatmap (matriz de correlación)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Facet grids
sns.FacetGrid(df, col='categoria').map(plt.hist, 'valor')
```

### Parámetros comunes
```python
# Tamaño y aspecto
plt.figure(figsize=(12, 8))

# Títulos y etiquetas
plt.title('Título del Gráfico')
plt.xlabel('Etiqueta X')
plt.ylabel('Etiqueta Y')

# Rotación de etiquetas
plt.xticks(rotation=45)

# Guardar
plt.savefig('grafico.png', dpi=300, bbox_inches='tight')

# Mostrar
plt.show()
```

## 🚀 TIPS Y TRUCOS ADICIONALES

### Cadenas de métodos (Method Chaining)
```python
resultado = (df
    .dropna()
    .groupby('categoria')
    .agg({'ventas': 'sum', 'cantidad': 'mean'})
    .sort_values('ventas', ascending=False)
    .head(10)
)
```

### Configuración de pandas
```python
# Mostrar más columnas/filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Formato de números
pd.set_option('display.float_format', '{:.2f}'.format)
```

### Funciones útiles para examen
```python
# Información rápida
df.sample(5)                    # 5 filas aleatorias
df.nlargest(5, 'columna')      # 5 valores más grandes
df.nsmallest(5, 'columna')     # 5 valores más pequeños
df.select_dtypes(include='number')  # Solo columnas numéricas
df.select_dtypes(include='object')  # Solo columnas de texto

# Estadísticas rápidas
df.corr()                      # Matriz de correlación
df.cov()                       # Matriz de covarianza
df['col'].quantile([0.25, 0.5, 0.75])  # Cuartiles específicos
```

---
*Esta guía cubre las operaciones más comunes de pandas para análisis exploratorio de datos. Úsala como referencia rápida.*



# Operaciones Adicionales para la Guía de Pandas

## 🔍 ANÁLISIS ESTADÍSTICO AVANZADO

### Detección de outliers
```python
# Método IQR (Rango Intercuartílico)
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['columna'] < Q1 - 1.5*IQR) | (df['columna'] > Q3 + 1.5*IQR)]

# Método Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['columna']))
outliers = df[z_scores > 3]

# Con seaborn para visualizar
sns.boxplot(y=df['columna'])  # Los puntos fuera de los bigotes son outliers
```

### Estadísticas de sesgo y curtosis
```python
from scipy.stats import skew, kurtosis

# Sesgo (skewness)
sesgo = df['columna'].skew()
sesgo_scipy = skew(df['columna'])

# Curtosis
curtosis = df['columna'].kurtosis()
curtosis_scipy = kurtosis(df['columna'])

# Para todo el DataFrame
df.skew()  # Sesgo de todas las columnas numéricas
df.kurtosis()  # Curtosis de todas las columnas numéricas
```

## 🔤 OPERACIONES CON STRINGS (str accessor)

### Manipulación de texto
```python
# Básicas
df['texto'].str.len()              # Longitud de cada string
df['texto'].str.lower()            # Minúsculas
df['texto'].str.upper()            # Mayúsculas
df['texto'].str.title()            # Primera letra mayúscula
df['texto'].str.strip()            # Eliminar espacios al inicio/final

# Búsqueda y reemplazo
df['email'].str.contains('@gmail')  # Contiene patrón
df['texto'].str.startswith('Mr.')   # Empieza con
df['texto'].str.endswith('.com')    # Termina con
df['texto'].str.replace('viejo', 'nuevo')  # Reemplazar
df['texto'].str.extract(r'(\d+)')   # Extraer con regex (primer grupo)

# División y acceso
df['nombre_completo'].str.split(' ')  # Dividir en lista
df['nombre_completo'].str.split(' ', expand=True)  # En columnas separadas
df['nombre_completo'].str[0]        # Primer carácter
df['nombre_completo'].str[0:5]      # Primeros 5 caracteres
```

## 📊 WINDOW FUNCTIONS (Funciones de ventana)

### Rolling (ventanas móviles)
```python
# Media móvil
df['media_movil_3'] = df['ventas'].rolling(window=3).mean()
df['media_movil_7'] = df['ventas'].rolling(window=7).mean()

# Otras funciones rolling
df['suma_movil'] = df['ventas'].rolling(window=5).sum()
df['std_movil'] = df['ventas'].rolling(window=10).std()
df['max_movil'] = df['ventas'].rolling(window=3).max()

# Con groupby
df['media_movil_por_categoria'] = df.groupby('categoria')['ventas'].rolling(window=3).mean()
```

### Expanding (ventanas expandibles)
```python
# Desde el inicio hasta cada punto
df['suma_acumulada'] = df['ventas'].expanding().sum()
df['media_acumulada'] = df['ventas'].expanding().mean()
```

### Shift y diff
```python
# Desplazar valores
df['ventas_anterior'] = df['ventas'].shift(1)    # Período anterior
df['ventas_siguiente'] = df['ventas'].shift(-1)   # Período siguiente

# Diferencias
df['cambio_ventas'] = df['ventas'].diff()         # Diferencia con período anterior
df['cambio_porcentual'] = df['ventas'].pct_change()  # Cambio porcentual
```

## 🎯 RANK Y PERCENTILES

```python
# Ranking
df['rank_ventas'] = df['ventas'].rank()                    # Ranking ascendente
df['rank_desc'] = df['ventas'].rank(ascending=False)       # Ranking descendente
df['rank_por_grupo'] = df.groupby('categoria')['ventas'].rank()

# Percentiles
df['percentil'] = df['ventas'].rank(pct=True)  # Percentil (0-1)
df['percentil_100'] = df['ventas'].rank(pct=True) * 100  # Percentil (0-100)
```

## 🔍 SAMPLING Y SELECCIÓN ALEATORIA

```python
# Muestra aleatoria
df.sample(n=100)                    # 100 filas aleatorias
df.sample(frac=0.1)                 # 10% de las filas
df.sample(n=50, random_state=42)    # Reproducible

# Muestra estratificada
df.groupby('categoria').apply(lambda x: x.sample(n=10))  # 10 de cada categoría
df.groupby('categoria').apply(lambda x: x.sample(frac=0.1))  # 10% de cada categoría
```

## 📋 VALIDACIÓN Y TESTING

### Assertions (verificaciones)
```python
# Verificar que no hay duplicados
assert df.duplicated().sum() == 0, "¡Hay filas duplicadas!"

# Verificar rangos de valores
assert (df['edad'] >= 0).all(), "¡Hay edades negativas!"
assert (df['edad'] <= 120).all(), "¡Hay edades irreales!"

# Verificar tipos de datos
assert df['fecha'].dtype == 'datetime64[ns]', "La columna fecha no es datetime"

# Verificar valores únicos
assert df['id'].nunique() == len(df), "¡Los IDs no son únicos!"
```

### Comparación de DataFrames
```python
# Comparar si son iguales
df1.equals(df2)

# Ver diferencias
pd.testing.assert_frame_equal(df1, df2)  # Error si son diferentes

# Comparar formas
assert df1.shape == df2.shape, "Los DataFrames tienen formas diferentes"
```

## 🚀 PERFORMANCE Y OPTIMIZACIÓN

### Reducir uso de memoria
```python
# Verificar uso de memoria
df.memory_usage(deep=True)

# Optimizar tipos de datos automáticamente
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except:
                pass
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df_optimized = optimize_dtypes(df.copy())

# Usar category para strings repetitivos
df['categoria'] = df['categoria'].astype('category')
```

### Vectorización vs loops
```python
# ❌ Evitar (lento)
for i, row in df.iterrows():
    df.loc[i, 'nueva_col'] = row['col1'] * row['col2']

# ✅ Preferir (rápido)
df['nueva_col'] = df['col1'] * df['col2']

# ✅ Con numpy para operaciones complejas
df['nueva_col'] = np.where(df['edad'] > 30, df['salario'] * 1.1, df['salario'])
```

## 📈 ANÁLISIS DE CORRELACIÓN AVANZADO

```python
# Matriz de correlación con diferentes métodos
corr_pearson = df.corr(method='pearson')    # Lineal (default)
corr_spearman = df.corr(method='spearman')  # Monotónica
corr_kendall = df.corr(method='kendall')    # Tau de Kendall

# Encontrar correlaciones altas
def find_high_corr(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    high_corr = corr_matrix.where(
        (corr_matrix > threshold) & (corr_matrix < 1.0)
    ).stack().dropna()
    return high_corr

correlaciones_altas = find_high_corr(df, 0.7)

# Visualizar correlaciones
import matplotlib.pyplot as plt
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, mask=mask, cmap='RdBu_r', center=0)
```

## 🎲 CROSS TABULATION Y CHI-CUADRADO

```python
# Tabla de contingencia
tabla_contingencia = pd.crosstab(df['categoria'], df['region'])
tabla_contingencia_prop = pd.crosstab(df['categoria'], df['region'], normalize='columns')

# Test Chi-cuadrado
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(tabla_contingencia)
print(f"Chi2: {chi2}, p-value: {p_value}")

# Con múltiples variables
pd.crosstab([df['categoria'], df['sexo']], df['region'])
```

## 📊 VISUALIZACIONES ADICIONALES CON SEABORN

```python
# Gráficos categóricos avanzados
sns.catplot(x='categoria', y='valor', hue='region', kind='box', data=df)
sns.catplot(x='categoria', y='valor', kind='violin', data=df)
sns.catplot(x='categoria', y='valor', kind='swarm', data=df)

# Distribuciones conjuntas
sns.jointplot(x='var1', y='var2', data=df, kind='reg')
sns.jointplot(x='var1', y='var2', data=df, kind='hex')

# Clustermap (con clustering jerárquico)
sns.clustermap(df.corr(), annot=True, cmap='RdBu_r', center=0)
```

## 🛠️ FUNCIONES ÚTILES ADICIONALES

### Información del sistema
```python
# Información de pandas
pd.show_versions()
pd.__version__

# Configuraciones actuales
pd.describe_option()
pd.get_option('display.max_rows')
pd.reset_option('all')  # Resetear todas las opciones
```

### Funciones de utilidad
```python
# Crear rangos de fechas
fechas = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
fechas_mensuales = pd.date_range(start='2023-01-01', periods=12, freq='M')

# Crear DataFrame con fechas
df_fechas = pd.DataFrame({'fecha': fechas, 'valor': np.random.randn(len(fechas))})

# Factorizar (convertir categorías a números)
df['categoria_num'], categorias = pd.factorize(df['categoria'])

# Crear bins automáticamente
df['bins'] = pd.qcut(df['valor'], q=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])
```

## ⚠️ CASOS ESPECIALES Y ERRORES COMUNES

### Manejo de MultiIndex
```python
# Crear MultiIndex
df_multi = df.set_index(['categoria', 'region'])

# Acceder a datos
df_multi.loc[('A', 'Norte')]
df_multi.loc[('A', 'Norte'), 'ventas']

# Resetear índice
df_multi.reset_index()

# Intercambiar niveles
df_multi.swaplevel().sort_index()
```

### Problemas comunes y soluciones
```python
# SettingWithCopyWarning - usar .copy()
df_subset = df[df['edad'] > 30].copy()
df_subset['nueva_col'] = 'valor'

# Comparación con NaN
df['columna'].isna()  # ✅ Correcto
df['columna'] == np.nan  # ❌ Siempre False

# Chained indexing
df.loc[df['edad'] > 30, 'categoria'] = 'Adulto'  # ✅ Correcto
df[df['edad'] > 30]['categoria'] = 'Adulto'  # ❌ Puede dar warning
```