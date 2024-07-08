# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:20:32 2024

@author: Windows11
"""

#from time import time
#from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib.ticker import FuncFormatter
#from matplotlib.ticker import MaxNLocator
from collections import Counter

# Cargo las tablas en DataFrame de pandas
df_works = pd.read_csv('C:/Users/Windows11/Desktop/intro-cd-gaston-martin-master-Tarea_1/Tarea_1/data/shakespeare/works.csv')
df_paragraphs = pd.read_csv('C:/Users/Windows11/Desktop/intro-cd-gaston-martin-master-Tarea_1/Tarea_1/data/shakespeare/paragraphs.csv')
df_chapters = pd.read_csv('C:/Users/Windows11/Desktop/intro-cd-gaston-martin-master-Tarea_1/Tarea_1/data/shakespeare/chapters.csv')
df_characters = pd.read_csv('C:/Users/Windows11/Desktop/intro-cd-gaston-martin-master-Tarea_1/Tarea_1/data/shakespeare/characters.csv')

# Visualizo el DataFrame de los trabajos realizados
df_works

# Agrupa los datos por 'character_id' y cuenta el número de párrafos por personaje
paragraphs_by_character = df_paragraphs.groupby('character_id').size().reset_index(name='num_paragraphs')

# Encuentra el personaje con más párrafos
max_paragraphs_character = paragraphs_by_character.loc[paragraphs_by_character['num_paragraphs'].idxmax()]

# Ordenar los personajes por el número de párrafos en orden descendente y seleccionar los primeros 5
top_5_characters = paragraphs_by_character.sort_values(by='num_paragraphs', ascending=False).head(5)

# Agrupar los datos por períodos de 5 años y género, y contar el número de obras
df_works['Period'] = (df_works['Date'] // 5) * 5
works_by_period_genre = df_works.groupby(['Period', 'GenreType']).size().unstack(fill_value=0).reset_index()

# Crear el gráfico de barras
works_by_period_genre.set_index('Period').plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Cantidad de Obras de Shakespeare por Períodos de 5 Años y Género')
plt.xlabel('Período')
plt.ylabel('Cantidad de Obras')
plt.xticks(rotation=45)
plt.legend(title='Género')


# Función para formatear los valores del eje Y como enteros
#def y_formatter(x, pos):
#    return f'{int(x)}'


df_paragraphs["PlainText"]

def clean_text(df, column_name):
    # Convertir todo a minúsculas
    result = df[column_name].str.lower()
    
# Quitar signos de puntuación y cambiarlos por espacios (" ")
    # TODO: completar signos de puntuación faltantes
    for punc in ["[", "\n", ",", ".", ";", ":", "?", "]", "--", "!", "(", ")"]:
        result = result.str.replace(punc, " ")
    return result    

# Creamos una nueva columna CleanText a partir de PlainText
df_paragraphs["CleanText"] = clean_text(df_paragraphs, "PlainText")

# Veamos la diferencia
df_paragraphs[["PlainText", "CleanText"]]

# Convierte párrafos en listas "palabra1 palabra2 palabra3" -> ["palabra1", "palabra2", "palabra3"]
df_paragraphs["WordList"] = df_paragraphs["CleanText"].str.split()

# Veamos la nueva columna creada
# Notar que a la derecha tenemos una lista: [palabra1, palabra2, palabra3]
df_paragraphs[["CleanText", "WordList"]]

# Nuevo dataframe: cada fila ya no es un párrafo, sino una sóla palabra
df_words = df_paragraphs.explode("WordList")

# Quitamos estas columnas redundantes
df_words.drop(columns=["CleanText", "PlainText"], inplace=True)

# Renombramos la columna WordList -> word
df_words.rename(columns={"WordList": "word"}, inplace=True)

# Verificar que el número de filas es mucho mayor
df_words

df_paragraphs

# Contar la frecuencia de cada palabra
word_counts = df_words['word'].value_counts().reset_index()
word_counts.columns = ['word', 'frequency']

# Mostrar las 10 palabras más frecuentes
top_words = word_counts.head(10)

# Crear el gráfico de barras para las palabras más frecuentes
plt.figure(figsize=(14, 8))
sns.barplot(x='frequency', y='word', data=top_words, palette='viridis')
plt.title('Las 10 Palabras Más Frecuentes en la Obra Completa de Shakespeare')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')


# Agregamos el nombre de los personajes
df_words = pd.merge(df_words, df_characters[["id", "CharName"]], left_on="character_id", right_on="id")

# Excluir el personaje "(stage directions)"
excluded_character = '(stage directions)'
df_words_filtered = df_words[df_words['CharName'] != excluded_character] 

words_per_character = df_words_filtered.groupby("CharName")["word"].count().sort_values(ascending=False)

words_per_character

char_show = words_per_character[:10]

plt.bar(char_show.index, char_show.values)
_ = plt.xticks(rotation=90)
