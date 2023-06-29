#Importamos las librerías que necesitaremos
from fastapi import FastAPI
import pandas as pd
from flask import Flask, request, jsonify
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Creamos una instancia de la aplicación FastAPI
app = FastAPI()

#Saludo, con esto confirmo que la API corre bien
@app.get('/')
def root():
    '''
        Esta función simplemente regesa un mensaje de confirmación, con esto podemos corroborar que
        nuestra API funciona correctamente
    '''
    return "ALFONSO JUSTO de DSPT-01 te dice, !Hola desde FastAPI!"
#Fin de nuestra función raíz *********************************************


# Función para obtener el número de películas que se estrenaron en algún mes
@app.get('/cantidad_filmaciones_mes/{mes}') #HECHO
def cantidad_filmaciones_mes(mes: str): #Definimos la función
    '''
        Como dice el primer renglón, con esta función recorremos el dataset específicamente en la columna donde 
        aparecen los meses de estreno de películas, identifica cuantas veces aparece el mes ingresado en la
        columna y nos regresa como resultado un mensaje indicando la cantidad de películas estrenadas en el mes
    '''

    #Creamos un manejador de errores, esto por si hay algún problema, en específico con la lectura del archivo
    try:
        #Leemos el archivo CSV y cargargamos los datos
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_ML_OPS\Dataset\limpios\movies_limpio (real) - copia.csv')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')

        #Contamos el número de veces que aparece el mes proporcionado por el usuario
        cantidad = df_movies['release_month'].str.lower().value_counts()[mes.lower()]
        '''
        Para tener un buen resultado, se convierten todos los valores de la columna en minúsculas, se cuenta
        la cantidad de veces que aparece el mes ingresado por el usuario y al final ese ingreso, se convierte
        a minisculas para que se pueda comparar correctamente con la columna
        '''
        
        #Regresamos la respuesta que será nuestro resultado
        return f"El mes de {mes} se tienen {cantidad} estrenos."
    
    except Exception as e:
        return f"Error al leer el archivo CSV o Error en el valor ingresado: {e}"
#Fin de nuestra función cantidad_filmaciones_mes *********************************************



# Función para obtener la cantidad de filmaciones en un día específico
@app.get('/cantidad_filmaciones_dia/{dia}') #HECHO
def cantidad_filmaciones_dia(dia: str):
    '''
        Con esta función, contamos cuantas películas se estrenaron en un día en específico, este lo
        proporciona el usuario
    '''
    
    try:
        #Leemos el archivo CSV y cargargamos los datos
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_ML_OPS\Dataset\limpios\movies_limpio (real) - copia.csv')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)

        #Contamos el número de veces que aparece el mes proporcionado por el usuario (lo mismo que la función anterior)
        cantidad_d = df_movies['release_day'].str.lower().value_counts()[dia.lower()]
        
        #Regresamos la respuesta que será nuestro resultado
        return f"En {dia} se tienen {cantidad_d} filmaciones estrenadas."
    
    except Exception as e:
        return f"Error al leer el archivo CSV o Error en el valor ingresado: {e}"
#Fin de nuestra función cantidad_filmaciones_dia *********************************************



# Función para obtener el título, año de estreno y score de una filmación
@app.get('/score_titulo/{titulo_de_la_filmacion}') #HECHO
def score_titulo(titulo_de_la_filmacion: str):
    '''
        Con esta función ingresamos el nombre de una película y lo que hará será ir a un dataset a buscar
        la información de la misma película, retornando el año de estreno y su popularidad
    '''
    
    try:
        #Leemos el archivo CSV y cargargamos los datos
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_ML_OPS\Dataset\limpios\movies_limpio (real) - copia.csv')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)

        #Filtramos el DataFrame por el título de la filmación ingresada por el usuario
        filtro = df_movies['title'].str.lower() == titulo_de_la_filmacion.lower()
        pelicula = df_movies.loc[filtro]
        
        #Obtenemos el título, el año de estreno y el score de la filmación
        titulo = pelicula['title'].iloc[0] #localizamos el titulo
        año_estreno = pelicula['release_year'].iloc[0] #localizamos el año
        score = pelicula['popularity'].iloc[0] #localizamos la popularidad, aquí tengo duda si es ese o 'vote_average'
        
        #Regresamos la respuesta que será nuestro resultado
        return f"La película {titulo} fue estrenada en el año {año_estreno} con un score/popularidad de {score}"
    except Exception as e:
        return f"Error al leer el archivo CSV o Error en el valor ingresado: {e}"
#Fin de nuestra función score_titulo *********************************************



# Función para obtener el título, cantidad de votos y valor promedio de una filmación
@app.get('/votos_titulo/{titulo_de_la_filmacion}') #HECHO
def votos_titulo(titulo_de_la_filmacion: str):
    try:
        #Leemos el archivo CSV y cargargamos los datos
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_ML_OPS\Dataset\limpios\movies_limpio (real) - copia.csv')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)
        
        #Filtramos nuestro dataset por el título de la filmación
        filtro = df_movies['title'].str.lower() == titulo_de_la_filmacion.lower()
        pelicula = df_movies.loc[filtro]
        
        #Obtenemos el título, la cantidad de votos y el valor promedio de las votaciones
        titulo = pelicula['title'].iloc[0]
        cantidad_votos = pelicula['vote_count'].iloc[0]
        promedio_votos = pelicula['vote_average'].iloc[0]
        
        #Probamos la condición como lo indica el enunciado
        if cantidad_votos >= 2000:
            return f"La película {titulo} cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votos}"
        else:
            return f"La película {titulo} no cumple con la condición de tener al menos 2000 valoraciones."
    except Exception as e:
        return f"Error al leer el archivo CSV o Error en el valor ingresado: {e}"
#Fin de nuestra función score_titulo *********************************************



# Función para obtener el éxito de un actor y detalles de su participación
@app.get('/get_actor/{nombre_actor}') #HECHO
def get_actor(nombre_actor: str):
    try:
        #Leemos el archivo CSV y cargargamos los datos
        #df_cast = pd.read_csv('F:/Henry/Proyecto 1/PI_ML_OPS/Dataset/limpios/cast_2 (con indice).csv', encoding='latin-1')
        #df_cast = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\cast_2 (con indice) - copia.csv', encoding='latin-1')
        enlace_cast = 'cast_2%20(con%20indice)%20-%20copia.csv'
        df_cast = pd.read_csv(enlace_cast)
        
        #Filtramos las filas con el nombre del actor ingresado
        actor_rows = df_cast[df_cast['name'].str.lower() == nombre_actor.lower()]
        
        #Revisamos si el actor está presente en el dataset, en caso de que no, lo menciona
        if actor_rows.empty:
            return f"No se encontró información para el actor {nombre_actor}"
        
        #Leemos el archivo 'movies.csv' para obtener el retorno y contar las películas
        #df_movies = pd.read_csv('F:/Henry/Proyecto 1/PI_ML_OPS/Dataset/limpios/movies_limpio (real) - copia.csv', encoding='latin-1')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)
        
        # Filtrar las filas de las películas en las que el actor participó por el ID del actor
        movies_rows = df_movies[df_movies['id'].isin(actor_rows['id'])]
        
        num_films = len(movies_rows)  # Número total de películas en las que participó el actor
        total_return = movies_rows['return'].sum()  # Retorno total del actor
        
        if num_films < 1:
            return f"El actor o la actriz {nombre_actor} no ha participado en ninguna película"
        
        avg_return = total_return / num_films  # Promedio de retorno
        max_return_movie = movies_rows.loc[movies_rows['return'].idxmax(), 'title']

        #Regresamos la respuesta que será nuestro resultado
        #Se agregó un dato más al resultado, y es la película que más retunr tiene
        #return f"El actor o la actriz {nombre_actor} ha participado en {num_films} filmaciones, con un retorno total de {total_return} y un promedio de {avg_return} por filmación"
        return f"El actor o la actriz {nombre_actor} ha participado en {num_films} filmaciones, con un retorno total de {total_return} y un promedio de {avg_return} por filmación. La película con el mayor retorno es: {max_return_movie}"

    except Exception as e:
        return f"Error al procesar la solicitud: {e}"
#Fin de nuestra función score_titulo *********************************************



# Función para obtener el éxito de un director y detalles de sus películas
@app.get('/get_director/{nombre_director}') #HECHO
def get_director(nombre_director: str):
    try:
        #Leemos el archivo CSV y cargargamos los datos
        #df_crew = pd.read_csv('F:/Henry/Proyecto 1/PI_ML_OPS/Dataset/limpios/crew_2 (con indice).csv', encoding='utf-8')
        #df_crew = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\crew_2 (con indice) - copia.csv', encoding='utf-8')
        enlace_crew = 'crew_2%20(con%20indice)%20-%20copia.csv'
        df_crew = pd.read_csv(enlace_crew)
        
        # Filtrar las filas del director deseado por nombre y trabajo como director
        director_rows = df_crew[(df_crew['name'] == nombre_director) & (df_crew['job'] == 'Director')]
        
        # Verificar si el director está presente en el dataset
        if director_rows.empty:
            return f"No se encontró información para el director {nombre_director}"

        #Leemos el siguiente archivo CSV y cargargamos los datos
        #df_movies = pd.read_csv('F:/Henry/Proyecto 1/PI_ML_OPS/Dataset/limpios/movies_limpio (real) - copia.csv', encoding='utf-8')
        #df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv', encoding='utf-8')
        enlace = 'movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
        df_movies = pd.read_csv(enlace)

        # Filtrar las filas de las películas del director por el ID del director
        director_films = df_movies[df_movies['id'].isin(director_rows['id'])]
        
        num_films = len(director_films)  # Número total de películas del director
        
        if num_films < 1:
            return f"El director {nombre_director} no ha dirigido ninguna película"
        
        total_return = director_films['return'].sum()  # Retorno total del director
        
        # Obtener los datos de cada película
        print('Estas son las películas que ha dirigido')
        films_data = []
        for index, film in director_films.iterrows():
            film_data = {
                'title': film['title'],
                'release_date': film['release_date'],
                'return': film['return'],
                'budget': film['budget'],
                'revenue': film['revenue']
            }
            films_data.append(film_data)
        
        return {
            'director_name': nombre_director,
            'num_films': num_films,
            'total_return': total_return,
            'films_data': films_data
        }
    
    except Exception as e:
        return f"Error al procesar la solicitud: {e}"
#Fin de nuestra función score_titulo *********************************************



# ML #HECHO
#Cargamos el conjunto de datos de películas adecuado
#movies_data = pd.read_csv('F:\Henry\Proyecto 1\PI_ML_OPS\Dataset\limpios\movies_limpio (real) - copia.csv')
movies_data = pd.read_csv('movies_limpio (real) - copia - copia.csv')
#enlace = 'https://github.com/AJusto14/PI_ML_OPS_AAJG/blob/main/Datasets/movies_limpio%20(real)%20-%20copia%20-%20copia.csv'
#movies_data = pd.read_csv(enlace)

#Realizamos el procesamiento de los datos
movies_data.fillna('', inplace=True)
movies_data['genres'] = movies_data.apply(lambda row: ' '.join(row['name_{}'.format(i)] for i in range(1, 7)), axis=1)
movies_data['overview'] = movies_data['overview'].str.lower()

#Creamos la matriz de características utilizando TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data['overview'])

#Comenzamos con la función
@app.get("/recomendador/{titulo}")
def recomendador(titulo: str):
    try:
        #Aquí inicialmente buscamos la película ingresada por el usuario
        movie_index = movies_data[movies_data['title'] == titulo].index
        if len(movie_index) == 0:
            raise HTTPException(status_code=404, detail="La película no existe en el dataset.") #representa excepciones relacionadas con solicitudes HTTP

        movie_index = movie_index[0]

        #Realizamos el cálculo de similitud de coseno entre la película ingresada y todas las demás
        similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

        #En esta parte obtenemos los índices de las películas más similares
        similar_movie_indices = similarity_scores.argsort()[::-1][1:6]

        #Tomamos los resultados y creamos una lista de recomendaciones
        recommendations = []
        for index in similar_movie_indices:
            movie = movies_data.iloc[index]
            recommendation = {
                'title': movie['title'],
                'genres': movie['name_1':'name_6'].tolist(),
                'popularity': movie['popularity'],
                'name_1_sl_1': movie['name_1_sl_1'],
                'release_date': movie['release_date'],
                'vote_average': movie['vote_average'],
                'vote_count': movie['vote_count'],
                'production_companies': movie['name_pc1':'name_pc21'].values.tolist(),
                'overview': movie['overview']
            }
            recommendations.append(recommendation)

        #Guardamos la información de la película ingresada por el usuario
        pelicula_ingresada = movies_data.loc[movie_index]
        info_pelicula_ingresada = {
            'Título': pelicula_ingresada['title'],
            'Géneros': pelicula_ingresada['genres'],
            'Popularidad': pelicula_ingresada['popularity'],
            'Idioma original': pelicula_ingresada['name_1_sl_1'],
            'Fecha de lanzamiento': pelicula_ingresada['release_date'],
            'Votación promedio': pelicula_ingresada['vote_average'],
            'Cantidad de votos': pelicula_ingresada['vote_count'],
            'Compañías productoras': pelicula_ingresada['name_pc1':'name_pc21'].values.tolist(),
            'Reseña': pelicula_ingresada['overview']
        }

        #Mostramos lo resultados, que es la información de la película ingresada y las recomendaciones
        return {
            'Película ingresada': info_pelicula_ingresada,
            'Recomendaciones': recommendations
        }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e)) 
#Fin de nuestra función recomendación *********************************************