# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:49:59 2024

@author: Windows11
"""

from time import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


"""
PARTE 1
"""

# Cargo las tablas en DataFrame de pandas
df_works = pd.read_csv('C:/Users/Windows11/Desktop/Latitud/MAESTRÍA/CURSOS/Introducción a la Ciencia de Datos/TAREA 1/intro-cd-gaston-martin-Tarea_1/Tarea_1/data/shakespeare/works.csv')
df_paragraphs = pd.read_csv('C:/Users/Windows11/Desktop/Latitud/MAESTRÍA/CURSOS/Introducción a la Ciencia de Datos/TAREA 1/intro-cd-gaston-martin-Tarea_1/Tarea_1/data/shakespeare/paragraphs.csv')
df_chapters = pd.read_csv('C:/Users/Windows11/Desktop/Latitud/MAESTRÍA/CURSOS/Introducción a la Ciencia de Datos/TAREA 1/intro-cd-gaston-martin-Tarea_1/Tarea_1/data/shakespeare/chapters.csv')
df_characters = pd.read_csv('C:/Users/Windows11/Desktop/Latitud/MAESTRÍA/CURSOS/Introducción a la Ciencia de Datos/TAREA 1/intro-cd-gaston-martin-Tarea_1/Tarea_1/data/shakespeare/characters.csv')

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

# Agregamos personajes, obras y géneros en el mismo dataset
df_dataset = df_paragraphs.merge(df_chapters.set_index("id")["work_id"], left_on="chapter_id", right_index=True)
df_dataset = df_dataset.merge(df_works.set_index("id")[["Title", "GenreType"]], left_on="work_id", right_index=True)
df_dataset = df_dataset.merge(df_characters.set_index('id')["CharName"], left_on="character_id", right_index=True).sort_index()
df_dataset = df_dataset[["CleanText", "CharName", "Title", "GenreType"]]

# Usaremos sólo estos personajes
characters = ["Antony", "Cleopatra", "Queen Margaret"]
df_dataset = df_dataset[df_dataset["CharName"].isin(characters)]

# Párrafos por cada personaje seleccionado
df_dataset["CharName"].value_counts()

X = df_dataset["CleanText"].to_numpy()
Y = df_dataset["CharName"].to_numpy()

# Partir train/test 30% estratificados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
print(f"Tamaños de Train/Test: {len(X_train)}/{len(X_test)}") 

# Contar la cantidad de párrafos por personaje en train y test
train_counts = pd.Series(Y_train).value_counts().sort_index()
test_counts = pd.Series(Y_test).value_counts().sort_index()

# Combinamos los conteos en un DataFrame
counts = pd.DataFrame({
    'train': train_counts,
    'test': test_counts
})

counts = counts.fillna(0)  # Llenar valores NaN con 0 si algún personaje no está en un conjunto

# Se genera el gráfico
counts.plot(kind='bar')
plt.xlabel('Character')
plt.ylabel('Number of Paragraphs')
plt.title('Balance of Paragraphs per Character in Train and Test Sets')
plt.xticks(rotation=0)
plt.show()

# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words=None, ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts

# Convertir la matriz sparse a una matriz densa para visualizar
print("Matriz de Unigramas (densa):\n", X_train_counts.toarray())

tf_idf = TfidfTransformer(use_idf=False)
X_train_tf = tf_idf.fit_transform(X_train_counts)
X_train_tf

from sklearn.decomposition import PCA

# Realizar PCA sobre los datos de entrenamiento, reducir la dimensionalidad a dos componentes principales
reductor = PCA(n_components=2)

# Transformar train
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Visualización de las dos primeras componentes de PCA
fig, ax = plt.subplots(figsize=(6, 6))
for character in np.unique(Y_train):
    mask_train = Y_train == character
    ax.scatter(X_train_red[mask_train, 0], X_train_red[mask_train, 1], label=character)

ax.set_title("PCA por personaje")
ax.legend()

"""
Realizamos algunos cambios y volvemos a generar la visualización
"""
# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts

tf_idf = TfidfTransformer(use_idf=True)
X_train_tf = tf_idf.fit_transform(X_train_counts)
X_train_tf

# Realizar PCA sobre los datos de entrenamiento, reducir la dimensionalidad a dos componentes principales
reductor = PCA(n_components=2)

# Transformar train
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Visualización de las dos primeras componentes de PCA
fig, ax = plt.subplots(figsize=(6, 6))
for character in np.unique(Y_train):
    mask_train = Y_train == character
    ax.scatter(X_train_red[mask_train, 0], X_train_red[mask_train, 1], label=character)

ax.set_title("PCA por personaje")
ax.legend()
"""
Lo anterior está duplicado ya que se implementaron algunas modificaciones en 
algunos parámetros
"""

# Visualización para ver cómo varía la varianza explicada a medida que se agregan componentes (hasta 10)
reductor = PCA(n_components=10)
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Calcular la varianza explicada y la varianza acumulativa
explained_variance = reductor.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Convertir a porcentaje
# explained_variance_percentage = explained_variance * 100
# cumulative_explained_variance_percentage = cumulative_explained_variance * 100

# Visualizar la varianza explicada y la varianza acumulativa
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), explained_variance, marker='o', label='Individual explained variance')
plt.plot(range(1, 11), cumulative_explained_variance, marker='o', label='Cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend()
plt.grid(True)
plt.show()


"""
PARTE 2
"""

# Modelos de clasificación
bayes_clf = MultinomialNB().fit(X_train_tf, Y_train)

# Ver las primeras 10 predicciones de train
Y_pred_train = bayes_clf.predict(X_train_tf)
Y_pred_train[:10]

def get_accuracy(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)

get_accuracy(Y_train, Y_pred_train)


# Vemos las predicciones sobre el conjunto test
X_test_counts = count_vect.transform(X_test)
X_test_tf = tf_idf.transform(X_test_counts)
Y_test_pred = bayes_clf.predict(X_test_tf)

get_accuracy(Y_test, Y_test_pred)


from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

accuracy = accuracy_score(Y_train, Y_pred_train) # Da el mismo valor que con la función get_accuracy
print("Accuracy:", accuracy)

# Visualizar la matriz de confusión utilizando ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_pred, display_labels=np.unique(Y), cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Reportar los valores de precision y recall para cada personaje
precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, Y_test_pred, labels=np.unique(Y))

report = pd.DataFrame({
    'Character': np.unique(Y),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score,
    'Support': support
})


# BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA (CROSS-VALIDATION)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

# TODO: Agregar más variantes de parámetros que les parezcan relevantes
param_sets = [{"stop_words": None, "ngram": (1,2), "idf": True},
              {"stop_words": None, "ngram": (1,2), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": True},
             {"stop_words": 'english', "ngram": (1,1), "idf": False},
             {"stop_words": 'english', "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": False}]

n_splits=4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Ahora usaremos train/validation/test
# Por lo tanto le renombramos train+validation = dev(elopment) dataset
X_dev = X_train
Y_dev = Y_train

# Almacenar los resultados
results = []

for params in param_sets:
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Transormaciones a aplicar (featurizers)
    count_vect = CountVectorizer(stop_words=params["stop_words"], ngram_range=params["ngram"])
    tf_idf = TfidfTransformer(use_idf=params["idf"])
    
    for train_idxs, val_idxs in skf.split(X_dev, Y_dev):
        
        # Train y validation para el split actual
        X_train_ = X_dev[train_idxs]
        Y_train_ = Y_dev[train_idxs]
        X_val = X_dev[val_idxs]
        Y_val = Y_dev[val_idxs]
        
        # Ajustamos y transformamos Train
        X_train_counts = count_vect.fit_transform(X_train_)
        X_train_tf = tf_idf.fit_transform(X_train_counts)
        
        # TODO: Completar el código para entrenar y evaluar 
        
        # Entrenamos con Train
        bayes_clf = MultinomialNB().fit(X_train_tf, Y_train_)
        
        # Transformamos Validation
        X_val_counts = count_vect.transform(X_val)
        X_val_tfidf = tf_idf.transform(X_val_counts)
        
        # Predecimos y evaluamos en Validation
        Y_pred_val = bayes_clf.predict(X_val_tfidf)
        acc = get_accuracy(Y_val, Y_pred_val)
        accuracies.append(acc)
        precisions.append(precision_score(Y_val, Y_pred_val, average='weighted'))
        recalls.append(recall_score(Y_val, Y_pred_val, average='weighted'))
        f1_scores.append(f1_score(Y_val, Y_pred_val, average='weighted'))
#        print(f"{acc=:.4f} {params=}")

    # Guardar resultados
    results.append({
        'params': params,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_f1_score': np.mean(f1_scores),
        'std_f1_score': np.std(f1_scores)})
    
# Mostrar los resultados
#for result in results:
#    print(f"Params: {result['params']}, Mean Accuracy: {result['mean_accuracy']:.4f}, Std Dev: {result['std_accuracy']:.4f}")

# Preparar datos para visualización
# Crear un DataFrame para los resultados
data = []
for result in results:
    for i in range(n_splits):
        data.append({
            'params': str(result['params']),
            'accuracy': result['accuracies'][i],
            'precision': result['precisions'][i],
            'recall': result['recalls'][i],
            'f1_score': result['f1_scores'][i]
        })


df = pd.DataFrame(data)

# Crear los gráficos de violín
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=metric, y='params', data=df, inner='point', scale='width')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Distribución de {metric.capitalize()} para cada Combinación de Parámetros', fontsize=14)
    plt.xlabel(metric.capitalize(), fontsize=16)
    plt.ylabel('Combinación de Parámetros', fontsize=16)
    plt.show()


"""
SELECCIONO LOS PARÁMETROS:
    stop_words = "english"
    n_gram = (1, 1)
    idf = True

Con estos parámetros vuelvo a entrenar sobre todo el conjunto de entrenamiento
(sin quitar datos para validación)
"""

# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)

tf_idf = TfidfTransformer(use_idf=True)
X_train_tf = tf_idf.fit_transform(X_train_counts)

# Realizar PCA sobre los datos de entrenamiento, reducir la dimensionalidad a dos componentes principales
reductor = PCA(n_components=2)

# Transformar train
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Visualización de las dos primeras componentes de PCA
fig, ax = plt.subplots(figsize=(6, 6))
for character in np.unique(Y_train):
    mask_train = Y_train == character
    ax.scatter(X_train_red[mask_train, 0], X_train_red[mask_train, 1], label=character)

ax.set_title("PCA por personaje")
ax.legend()

# Visualización para ver cómo varía la varianza explicada a medida que se agregan componentes (hasta 10)
reductor = PCA(n_components=10)
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Calcular la varianza explicada y la varianza acumulativa
explained_variance = reductor.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Convertir a porcentaje
explained_variance_percentage = explained_variance * 100
cumulative_explained_variance_percentage = cumulative_explained_variance * 100

# Visualizar la varianza explicada y la varianza acumulativa
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), explained_variance_percentage, marker='o', label='Individual explained variance')
plt.plot(range(1, 11), cumulative_explained_variance_percentage, marker='o', label='Cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend()
plt.grid(True)
plt.show()

# Modelos de clasificación
bayes_clf = MultinomialNB().fit(X_train_tf, Y_train)

# Vemos las predicciones sobre el conjunto test
X_test_counts = count_vect.transform(X_test)
X_test_tf = tf_idf.transform(X_test_counts)
Y_test_pred = bayes_clf.predict(X_test_tf)

get_accuracy(Y_test, Y_test_pred)

# Visualizar la matriz de confusión utilizando ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_pred, display_labels=np.unique(Y), cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Reportar los valores de precision y recall para cada personaje
precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, Y_test_pred, labels=np.unique(Y))

report = pd.DataFrame({
    'Character': np.unique(Y),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score,
    'Support': support
})


# VAMOS A UTILIZAR OTRO MODELO PARA CLASIFICAR TEXTO
# Distinto de Multinomial Naive Bayes, se llama Support Vector Machine (SVM)

# BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA (CROSS-VALIDATION)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

# TODO: Agregar más variantes de parámetros que les parezcan relevantes
param_sets = [{"stop_words": None, "ngram": (1,2), "idf": True},
              {"stop_words": None, "ngram": (1,2), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": True},
             {"stop_words": 'english', "ngram": (1,1), "idf": False},
             {"stop_words": 'english', "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": False}]

n_splits=4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Ahora usaremos train/validation/test
# Por lo tanto le renombramos train+validation = dev(elopment) dataset
X_dev = X_train
Y_dev = Y_train

# Almacenar los resultados
results_svm = []

for params in param_sets:
    
    accuracies_svm = []
    precisions_svm = []
    recalls_svm = []
    f1_scores_svm = []
    
    # Transormaciones a aplicar (featurizers)
    count_vect = CountVectorizer(stop_words=params["stop_words"], ngram_range=params["ngram"])
    tf_idf = TfidfTransformer(use_idf=params["idf"])
    
    for train_idxs, val_idxs in skf.split(X_dev, Y_dev):
        
        # Train y validation para el split actual
        X_train_ = X_dev[train_idxs]
        Y_train_ = Y_dev[train_idxs]
        X_val = X_dev[val_idxs]
        Y_val = Y_dev[val_idxs]
        
        # Ajustamos y transformamos Train
        X_train_counts = count_vect.fit_transform(X_train_)
        X_train_tf = tf_idf.fit_transform(X_train_counts)
        
        # TODO: Completar el código para entrenar y evaluar 
        
        # Entrenamos con Train
        svm_model = LinearSVC(random_state=42)
        svm_model.fit(X_train_tf, Y_train_)
        
        # Transformamos Validation
        X_val_counts = count_vect.transform(X_val)
        X_val_tfidf = tf_idf.transform(X_val_counts)
        
        # Predecimos y evaluamos en Validation
        Y_pred_val = svm_model.predict(X_val_tfidf)
        acc = get_accuracy(Y_val, Y_pred_val)
        accuracies.append(acc)
        precisions.append(precision_score(Y_val, Y_pred_val, average='weighted'))
        recalls.append(recall_score(Y_val, Y_pred_val, average='weighted'))
        f1_scores.append(f1_score(Y_val, Y_pred_val, average='weighted'))

    # Guardar resultados
    results_svm.append({
        'params': params,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_f1_score': np.mean(f1_scores),
        'std_f1_score': np.std(f1_scores)})
    
# Preparar datos para visualización
# Crear un DataFrame para los resultados
data_svm = []
for result in results:
    for i in range(n_splits):
        data_svm.append({
            'params': str(result['params']),
            'accuracy': result['accuracies'][i],
            'precision': result['precisions'][i],
            'recall': result['recalls'][i],
            'f1_score': result['f1_scores'][i]
        })


df = pd.DataFrame(data_svm)

# Crear los gráficos de violín
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=metric, y='params', data=df, inner='point', scale='width')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Distribución de {metric.capitalize()} para cada Combinación de Parámetros', fontsize=14)
    plt.xlabel(metric.capitalize(), fontsize=16)
    plt.ylabel('Combinación de Parámetros', fontsize=16)
    plt.show()


"""
SELECCIONO LOS PARÁMETROS:
    stop_words = "english"
    n_gram = (1, 1)
    idf = True

Con estos parámetros vuelvo a entrenar sobre todo el conjunto de entrenamiento
(sin quitar datos para validación)
"""

# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)

tf_idf = TfidfTransformer(use_idf=True)
X_train_tf = tf_idf.fit_transform(X_train_counts)

# Modelos de clasificación
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tf, Y_train)

# Vemos las predicciones sobre el conjunto test
X_test_counts = count_vect.transform(X_test)
X_test_tf = tf_idf.transform(X_test_counts)
Y_test_pred = svm_model.predict(X_test_tf)

get_accuracy(Y_test, Y_test_pred)

# Visualizar la matriz de confusión utilizando ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_pred, display_labels=np.unique(Y), cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Reportar los valores de precision y recall para cada personaje
precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, Y_test_pred, labels=np.unique(Y))

report_svm = pd.DataFrame({
    'Character': np.unique(Y),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score,
    'Support': support
})


# CAMBIAMOS UN PERSONAJE Y REPETIMOS EL PROCEDIMIENTO UTILIZANDO BAYESIAN
# Cambiamos a Cleopatra por Falstaff

# Agregamos personajes, obras y géneros en el mismo dataset
df_dataset = df_paragraphs.merge(df_chapters.set_index("id")["work_id"], left_on="chapter_id", right_index=True)
df_dataset = df_dataset.merge(df_works.set_index("id")[["Title", "GenreType"]], left_on="work_id", right_index=True)
df_dataset = df_dataset.merge(df_characters.set_index('id')["CharName"], left_on="character_id", right_index=True).sort_index()
df_dataset = df_dataset[["CleanText", "CharName", "Title", "GenreType"]]

# Usaremos sólo estos personajes
characters = ["Falstaff", "Antony", "Queen Margaret"]
df_dataset = df_dataset[df_dataset["CharName"].isin(characters)]

# Párrafos por cada personaje seleccionado
df_dataset["CharName"].value_counts()

X = df_dataset["CleanText"].to_numpy()
Y = df_dataset["CharName"].to_numpy()

# Partir train/test 30% estratificados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
print(f"Tamaños de Train/Test: {len(X_train)}/{len(X_test)}") 

# Contar la cantidad de párrafos por personaje en train y test
train_counts = pd.Series(Y_train).value_counts().sort_index()
test_counts = pd.Series(Y_test).value_counts().sort_index()

# Combinamos los conteos en un DataFrame
counts = pd.DataFrame({
    'train': train_counts,
    'test': test_counts
})

counts = counts.fillna(0)  # Llenar valores NaN con 0 si algún personaje no está en un conjunto

# Se genera el gráfico
counts.plot(kind='bar')
plt.xlabel('Character')
plt.ylabel('Number of Paragraphs')
plt.title('Balance of Paragraphs per Character in Train and Test Sets')
plt.xticks(rotation=0)
plt.show()

# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)

tf_idf = TfidfTransformer(use_idf=True)
X_train_tf = tf_idf.fit_transform(X_train_counts)

# Realizar PCA sobre los datos de entrenamiento, reducir la dimensionalidad a dos componentes principales
reductor = PCA(n_components=2)

# Transformar train
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Visualización de las dos primeras componentes de PCA
fig, ax = plt.subplots(figsize=(6, 6))
for character in np.unique(Y_train):
    mask_train = Y_train == character
    ax.scatter(X_train_red[mask_train, 0], X_train_red[mask_train, 1], label=character)

ax.set_title("PCA por personaje")
ax.legend()

# Visualización para ver cómo varía la varianza explicada a medida que se agregan componentes (hasta 10)
reductor = PCA(n_components=10)
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Calcular la varianza explicada y la varianza acumulativa
explained_variance = reductor.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Convertir a porcentaje
explained_variance_percentage = explained_variance * 100
cumulative_explained_variance_percentage = cumulative_explained_variance * 100

# Visualizar la varianza explicada y la varianza acumulativa
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), explained_variance_percentage, marker='o', label='Individual explained variance')
plt.plot(range(1, 11), cumulative_explained_variance_percentage, marker='o', label='Cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend()
plt.grid(True)
plt.show()


"""
PARTE 2
"""

# BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA (CROSS-VALIDATION)

# TODO: Agregar más variantes de parámetros que les parezcan relevantes
param_sets = [{"stop_words": None, "ngram": (1,2), "idf": True},
              {"stop_words": None, "ngram": (1,2), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": False},
             {"stop_words": None, "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": True},
             {"stop_words": 'english', "ngram": (1,1), "idf": False},
             {"stop_words": 'english', "ngram": (1,1), "idf": True},
             {"stop_words": 'english', "ngram": (1,2), "idf": False}]

n_splits=4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Ahora usaremos train/validation/test
# Por lo tanto le renombramos train+validation = dev(elopment) dataset
X_dev = X_train
Y_dev = Y_train

# Almacenar los resultados
results = []

for params in param_sets:
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Transormaciones a aplicar (featurizers)
    count_vect = CountVectorizer(stop_words=params["stop_words"], ngram_range=params["ngram"])
    tf_idf = TfidfTransformer(use_idf=params["idf"])
    
    for train_idxs, val_idxs in skf.split(X_dev, Y_dev):
        
        # Train y validation para el split actual
        X_train_ = X_dev[train_idxs]
        Y_train_ = Y_dev[train_idxs]
        X_val = X_dev[val_idxs]
        Y_val = Y_dev[val_idxs]
        
        # Ajustamos y transformamos Train
        X_train_counts = count_vect.fit_transform(X_train_)
        X_train_tf = tf_idf.fit_transform(X_train_counts)
        
        # TODO: Completar el código para entrenar y evaluar 
        
        # Entrenamos con Train
        bayes_clf = MultinomialNB().fit(X_train_tf, Y_train_)
        
        # Transformamos Validation
        X_val_counts = count_vect.transform(X_val)
        X_val_tfidf = tf_idf.transform(X_val_counts)
        
        # Predecimos y evaluamos en Validation
        Y_pred_val = bayes_clf.predict(X_val_tfidf)
        acc = get_accuracy(Y_val, Y_pred_val)
        accuracies.append(acc)
        precisions.append(precision_score(Y_val, Y_pred_val, average='weighted'))
        recalls.append(recall_score(Y_val, Y_pred_val, average='weighted'))
#        f1_scores.append(f1_score(Y_val, Y_pred_val))

    # Guardar resultados
    results.append({
        'params': params,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
#        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls)})
#        'mean_f1_score': np.mean(f1_scores),
#        'std_f1_score': np.std(f1_scores)})
    
# Preparar datos para visualización
# Crear un DataFrame para los resultados
data = []
for result in results:
    for i in range(n_splits):
        data.append({
            'params': str(result['params']),
            'accuracy': result['accuracies'][i],
            'precision': result['precisions'][i],
            'recall': result['recalls'][i],
#            'f1_score': result['f1_scores'][i]
        })


df = pd.DataFrame(data)

# Crear los gráficos de violín
metrics = ['accuracy', 'precision', 'recall']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=metric, y='params', data=df, inner='point', scale='width')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Distribución de {metric.capitalize()} para cada Combinación de Parámetros', fontsize=14)
    plt.xlabel(metric.capitalize(), fontsize=16)
    plt.ylabel('Combinación de Parámetros', fontsize=16)
    plt.show()


"""
SELECCIONO LOS PARÁMETROS:
    stop_words = "english"
    n_gram = (1, 1)
    idf = True

Con estos parámetros vuelvo a entrenar sobre todo el conjunto de entrenamiento
(sin quitar datos para validación)
"""

# Conteo de palabras y TF-IDF
count_vect = CountVectorizer(stop_words='english', ngram_range=(1,1))
X_train_counts = count_vect.fit_transform(X_train)

tf_idf = TfidfTransformer(use_idf=False)
X_train_tf = tf_idf.fit_transform(X_train_counts)

# Realizar PCA sobre los datos de entrenamiento, reducir la dimensionalidad a dos componentes principales
reductor = PCA(n_components=2)

# Transformar train
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Visualización de las dos primeras componentes de PCA
fig, ax = plt.subplots(figsize=(6, 6))
for character in np.unique(Y_train):
    mask_train = Y_train == character
    ax.scatter(X_train_red[mask_train, 0], X_train_red[mask_train, 1], label=character)

ax.set_title("PCA por personaje")
ax.legend()

# Visualización para ver cómo varía la varianza explicada a medida que se agregan componentes (hasta 10)
reductor = PCA(n_components=10)
X_train_red = reductor.fit_transform(X_train_tf.toarray()) 

# Calcular la varianza explicada y la varianza acumulativa
explained_variance = reductor.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Convertir a porcentaje
explained_variance_percentage = explained_variance * 100
cumulative_explained_variance_percentage = cumulative_explained_variance * 100

# Visualizar la varianza explicada y la varianza acumulativa
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), explained_variance_percentage, marker='o', label='Individual explained variance')
plt.plot(range(1, 11), cumulative_explained_variance_percentage, marker='o', label='Cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.legend()
plt.grid(True)
plt.show()

# Modelos de clasificación
bayes_clf = MultinomialNB().fit(X_train_tf, Y_train)

# Vemos las predicciones sobre el conjunto test
X_test_counts = count_vect.transform(X_test)
X_test_tf = tf_idf.transform(X_test_counts)
Y_test_pred = bayes_clf.predict(X_test_tf)

get_accuracy(Y_test, Y_test_pred)

# Visualizar la matriz de confusión utilizando ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_pred, display_labels=np.unique(Y), cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Reportar los valores de precision y recall para cada personaje
precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, Y_test_pred, labels=np.unique(Y))

report = pd.DataFrame({
    'Character': np.unique(Y),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score,
    'Support': support
})














