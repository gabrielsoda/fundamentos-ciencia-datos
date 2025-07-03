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