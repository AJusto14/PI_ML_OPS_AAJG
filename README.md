# <h1 align=center> **PROYECTO INDIVIDUAL Nº1 - Alfonso Justo** </h1>
# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>
# <h1 align=center>**`Nombre del Proyecto: PI_ML_OPS_AAJG`**</h1>
# 
## **Descripción del problema (Contexto y rol a desarrollar)**

## Contexto

Tienes tu modelo de recomendación dando unas buenas métricas, y ahora, cómo lo llevas al mundo real?

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.

<br>

## Rol a desarrollar

Empezaste a trabajar como **`Data Scientist`** en una start-up que provee servicios de agregación de plataformas de streaming. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: un sistema de recomendación que aún no ha sido puesto en marcha! 

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula :sob:): Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevas películas o series, entre otras cosas….  haciendo tu trabajo imposible :weary:. 

Debes empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** (_Minimum Viable Product_) para la próxima semana! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir :exclamation:. Así que te espantas los miedos y te pones manos a la obra :muscle:

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

<sub> Nota que aqui se reflejan procesos no herramientas tecnologicas. Has el ejercicio de entender cual herramienta del stack corresponde a cual parte del proceso<sub/>
<br>

## **Propuesta de trabajo (requerimientos de aprobación)**

**`Transformaciones`**:  Para este MVP no necesitas perfección, ¡necesitas rapidez! ⏩ Vas a hacer las transformaciones, ***y solo estas*** indicadas.
<br/>
<br>

**`Desarrollo API`**:   Propones disponibilizar los datos de la empresa usando el framework ***FastAPI***. Las consultas son las que están en las instrucciones.

Deben crear 6 funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).


<br/>


**`Deployment`**: Conoces sobre [Render](https://render.com/docs/free#free-web-services) y tienes un [tutorial de Render](https://github.com/HX-FNegrete/render-fastapi-tutorial) que te hace la vida mas facil :smile: . Tambien podrias usar [Railway](https://railway.app/), o cualquier otro servicio que permita que la API pueda ser consumida desde la web.

<br/>

**`Análisis exploratorio de los datos`**: _(Exploratory Data Analysis-EDA)_

Se derberá realizar en los datasets proporcionados.

<br>

**`Sistema de recomendación`**: 

Éste consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score de similaridad y devolverá una lista de Python con 5 valores, cada uno siendo el string del nombre de las películas con mayor puntaje, en orden descendente. Debe ser deployado como una función adicional de la API anterior y debe llamarse:


+ def **recomendacion( *`titulo`* )**:
    Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

<br/>

**`Video`**: Necesitas que al equipo le quede claro que tus herramientas funcionan realmente! Haces un video mostrando el resultado de las consultas propuestas y de tu modelo de ML entrenado!

<sub> **Spoiler**: El video NO DEBE durar mas de ***7 minutos*** y DEBE mostrar las consultas requeridas en funcionamiento desde la API y una breve explicacion del modelo utilizado para el sistema de recomendacion. En caso de que te sobre tiempo luego de grabarlo, puedes mostrar explicar tu EDA, ETL e incluso cómo desarrollaste la API. <sub/>

<br/>

<br>

# SOLUCIONES
<br>

### TRANSFORMACIONES
<br>
La realización de las transformaciones las pueden encontrar en el archivo llamado 'limpieza_movies.ipynb'.
<br>
En el aparte de los puntos solicitados, encontrarán un poco de la explicación de porque se hicieron las cosas de esa forma.
<br>
Cada una de las transformaciones se realizó en Python, con excepción de la primera, que la justificación la encontrarán en el archivo.
<br>
<br>
Imagen de un hallazgo (imagen de la gráfica).
<br>
<p align="center">
<img src="https://raw.githubusercontent.com/AJusto14/PI_ML_OPS_AAJG/tree/main/src/imagen_transformaciones.png"  height=500>
</p>
<br>

## *Eliminar las columnas que no serán utilizadas, video, imdb_id, adult, original_title, poster_path y homepage.*.
<br>

```python
#Ahora solo dejamos las columnas con las que vamos a trabajar
pelis = pelis[['belongs_to_collection', 'budget', 'genres', 'id', 'original_language', 'overview',
       'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'vote_average', 'vote_count']]
``` 
<br>
<br>

### DESARROLLO DE LA API
<br>
El desarrollo de la API se llevó a cabo con la librería FASTAPI, que al no tenerla instalada, de inicio se tuvo que realizar la instalación para poder manejar correctamente lo solicitado.
<br>
Afortunadamente se lograron cumplir con los puntos solicitados, con todos y cada uno de ellos.
<br>
<br>
Aquí un ejemplo de las consigas solicitadas:
<br>
<br>

## *- def cantidad_filmaciones_mes( Mes ): Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset*.
Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X.*.
<br>

```python
# Función para obtener el número de películas que se estrenaron en algún mes
@app.get('/cantidad_filmaciones_mes/{mes}') #HECHO
def cantidad_filmaciones_mes(mes: str): #Definimos la función
    
    #Creamos un manejador de errores, esto por si hay algún problema, en específico con la lectura del archivo
    try:
        #Leemos el archivo CSV
        df_movies = pd.read_csv('F:\Henry\Proyecto 1\PI_1_Henry_MachineLearningDevops\PI_ML_OPS_AAJG\Datasets\movies_limpio (real) - copia - copia.csv')

        #Contamos el número de veces que aparece el mes proporcionado por el usuario
        cantidad = df_movies['release_month'].str.lower().value_counts()[mes.lower()]
        
        #Regresamos la respuesta que será nuestro resultado
        return f"El mes de {mes} se tienen {cantidad} estrenos."
    
    except Exception as e:
        return f"Error al leer el archivo CSV o Error en el valor ingresado: {e}"
#Fin de nuestra función cantidad_filmaciones_mes *********************************************
``` 
<br>
Imagen de la vista en la API (captura de la api funcionando, el saludo).
<br>
<p align="center">
<img src="https://raw.githubusercontent.com/AJusto14/PI_ML_OPS_AAJG/tree/main/src/api_funcionando.png"  height=500>
</p> 
<br>

### DEPLOYMENT
<br>
Pendiente

<br>
<br>

### EXPLORATORY DATA ANALYSIS - EDA
<br>
Como sabemos el análisis EDA es una parte importante de nuestra actividad como Científicos de Datos, y para esto, en lo personal me guié con las siguientes pautas:<br><br>
- Comprensión de los datos<br>
- Herramientas necesarias (librerías, entorno, etc.)<br>
- Datos (en nuestro caso, los datasets)<br>
- Exploración Básica (info(), describe())<br>
- Limpieza (eliminar nulos, NaN, etc.)<br>
- Transformaciones necesarias (variables con valores correctos)<br>
- Visualizaciones (gráficos exploratorios)<br>
- Análisis Estadísticos Simples (media, moda, mediana, etc.)<br>
- Extracción de Información Relevante (filtros, consultas)<br>
<br>
Se realizaron todas las pautas mencionadas, se localizó información interesante.<br>
Aquí un ejemplo de lo que se localizó.
<br>
<br>

```python
#Ahora solo dejamos las columnas con las que vamos a trabajar
pelis = pelis[['belongs_to_collection', 'budget', 'genres', 'id', 'original_language', 'overview',
       'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'vote_average', 'vote_count']]
``` 
<br>
<br>

### SISTEMA DE RECOMENDACIÓN
<br>
Para la realización de esta parte, se utilizó un algoritmo que se conoce como "sistema de recomendación basado en contenido". Se decidió esto después de investigar un poco más al respecto, y se concluyó que este algoritmo al utilizar la "similitud de coseno" para el cálculo de la similitud entre las películas en función de sus características de contenido, sería perfecto para lo que necesitaba. <br>
Ademas, con este algorimo recibimos recomiendaciones de las películas más similares a la ingresada por el usuario.<br>
Este enfoque se utiliza porque permite proporcionar recomendaciones personalizadas basadas en las características del contenido de las películas, en nuestro caso, tomamos la mayoría de las columnas con las que contaba el dataset para realizar la predicción. <br>
Incluso, al utilizar TF-IDF, se puede identificar la importancia de las categorías y demás información y con eso poder encontrar películas que tengan similitudes con la película que el usuario ingresó.<br>
<br>
Pero, quizá se pregunten, porque no se eligió un algoritmo visto en clase, por ejemplo alguno de clasificación, y esto es sencillo, debido a que tenemos tres caracteristicas importantes:
<br><br>

1. *Disponibilidad de información*: El enfoque basado en contenido utiliza las características y atributos de las películas, como género, sinopsis, actores, etc. Estos datos suelen estar fácilmente disponibles en los conjuntos de datos de películas. Por otro lado, un enfoque de clasificación requiere etiquetas explícitas de preferencia o clasificación de películas por parte de los usuarios, lo cual puede ser más difícil de obtener o menos confiable.

2. *Escalabilidad*: El enfoque basado en contenido puede ser más escalable en términos de rendimiento y tiempo de respuesta, ya que el cálculo de similitud se realiza utilizando características preexistentes de las películas. Por el contrario, un enfoque de clasificación podría requerir un proceso de entrenamiento más complejo y costoso, especialmente si se tiene un gran número de películas y usuarios.

3. *Interpretación de resultados*: En el enfoque basado en contenido, las recomendaciones se pueden explicar fácilmente al usuario en términos de las características de las películas que influyeron en la recomendación. Esto puede ayudar a generar confianza y comprensión por parte del usuario, ya que las recomendaciones se basan en sus preferencias de género, actores favoritos, etc. En el caso de un enfoque de clasificación, puede ser más difícil explicar las recomendaciones en términos de las razones específicas detrás de ellas.

<br>
Este fue mi resultado de la función: recomendacion/ (imagen del resultado)
<br>
<br>
<p align="center">
<img src="https://raw.githubusercontent.com/AJusto14/PI_ML_OPS_AAJG/tree/main/src/resultado_recomendacion.png"  height=500>
</p>
<br>
<br>

### VIDEO
<br>
Este es el enlace del video de demostración del funcionamiento:
