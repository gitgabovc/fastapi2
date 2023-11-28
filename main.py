from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/playtimegenre/{genre}")
# def read_playtime_genre(genre: str):
#     df_steam_1 = pd.read_parquet("steam_games_clean.parquet")
#     df_user_items_1 = pd.read_parquet("user_items_clean.parquet")
#     df_combinado = pd.merge(df_user_items_1 ,df_steam_1, on='item_id', how='inner')
#     df_agrupado = (
#         df_combinado[['release_date', 'playtime_forever', genre]]
#         .query(f"`{genre}` == 1")
#         .groupby('release_date')['playtime_forever']
#         .sum()
#         .reset_index()
#     )

#     # Encontrar el año más jugado
#     anio_mas_jugado = df_agrupado.loc[df_agrupado['playtime_forever'].idxmax()]

#     return {"releasse_date": int(anio_mas_jugado['release_date'])}

# Lee la tabla pivot desde el archivo Parquet

@app.get("/playtimegenre/{genre}")
def get_max_playtime(genre: str):
    tabla_pivot_steam_user_items = pd.read_parquet("Dataset/tabla_pivot_steam_user_items.parquet")
    # Filtra la columna del género específico
    genre_column = tabla_pivot_steam_user_items[genre]

    # Encuentra el año con el tiempo de juego máximo
    max_playtime_year = genre_column.idxmax()

    return {"Año de lanzamiento con más horas jugadas para Género "+ genre: int(max_playtime_year)}



@app.get('/userforgenre/{genre}')
def read_user_for_genre(genre: str):
    # Lee los archivos CSV y combina los DataFrames
    df_filtrado = (
        pd.merge(
            pd.read_parquet("user_items_clean.parquet"),
            pd.read_parquet("steam_games_clean.parquet"),
            on='item_id',
            how='inner'
        )
        .loc[lambda df: df[genre] == 1, ['release_date', 'playtime_forever', 'user_id', genre]]
    )

    # Encontrar el jugador con más horas
    jugador_mas_horas_id = df_filtrado.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Filtrar directamente el DataFrame por el jugador con más horas y calcular las horas por año
    df_agrupado_release_anio = (
        df_filtrado[df_filtrado['user_id'] == jugador_mas_horas_id]
        .groupby('release_date')['playtime_forever']
        .sum()
        .reset_index()
    )

    # Crear la estructura de respuesta con nombres de columnas modificados
    respuesta = {
        "Usuario con más horas jugadas para Género " + genre: jugador_mas_horas_id,
        "Horas jugadas": df_agrupado_release_anio.rename(columns={"release_date": "Anio", "playtime_forever": "Horas"}).to_dict(orient='records')
    }

    # Devolver la respuesta como JSON
    return JSONResponse(content=respuesta)

@app.get("/usersrecommend/{anio}")
def read_users_recommend(anio: int):

    
    df_steam_1 = pd.read_parquet("steam_games_clean.parquet")
    df_user_reviews = pd.read_parquet("user_reviews_clean.parquet")
    df = pd.merge(df_user_reviews ,df_steam_1, on='item_id', how='inner')

    # Filtrar y encadenar las operaciones
    top3_apps = (
        df[df['anio'] == anio]
        .loc[df['recommend'] & df['sentiment_analysis'].isin([1, 2]), ['app_name', 'sentiment_analysis']]
        .groupby('app_name')['sentiment_analysis']
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    # Crear la estructura de respuesta
    respuesta = [{"Puesto {}: ".format(i + 1): app} for i, app in enumerate(top3_apps)]

    # Devolver la respuesta como JSON
    return JSONResponse(content=respuesta)


@app.get("/usersworstdeveloper/{anio}")
def read_usersworstdeveloper(anio: int):
    df_steam_1 = pd.read_parquet("steam_games_clean.parquet")
    df_user_reviews = pd.read_parquet("user_reviews_clean.parquet")
    df = pd.merge(df_user_reviews ,df_steam_1, on='item_id', how='inner')
    
    dre = (
        df[df['anio'] == anio]
        .loc[lambda x: x['recommend'] == False]
        .loc[lambda x: x['sentiment_analysis'] == 0]
        .groupby('developer')
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
        .head(3)
    )
    
    # Crear la estructura de respuesta solo con el nombre del desarrollador
    respuesta = [{"Puesto {}: ".format(i + 1): developer} for i, developer in enumerate(dre['developer'])]

    # Devolver la respuesta como JSON
    return JSONResponse(content=respuesta)




@app.get("/sentimentanalysis/{developer}")
def read_sentiment_analysis(developer: str):
    
    
    df_steam_1 = pd.read_parquet("steam_games_clean.parquet")
    df_user_reviews = pd.read_parquet("user_reviews_clean.parquet")
    df = pd.merge(df_user_reviews ,df_steam_1, on='item_id', how='inner')
    
    # Filtrar por el desarrollador específico
    df_filtrado = df[df['developer'] == developer]
    
    # Contar las ocurrencias y rellenar con 0
    resultados = (
        pd.DataFrame({'sentiment_analysis': [0, 1, 2]})
        .merge(df_filtrado.groupby('sentiment_analysis').size().reset_index(name='count'), 
               how='left', on='sentiment_analysis')
        .fillna(0)
        .astype({'count': int})
    )
    
    # Crear la estructura de respuesta directamente
    respuesta = {
        developer: {
            "Negative": int(resultados.loc[resultados['sentiment_analysis'] == 0, 'count']),
            "Neutral": int(resultados.loc[resultados['sentiment_analysis'] == 1, 'count']),
            "Positive": int(resultados.loc[resultados['sentiment_analysis'] == 2, 'count'])
        }
    }

    # Devolver la respuesta como JSON
    return JSONResponse(content=respuesta)


@app.get("/recomendacionjuego/{id_juego_conocido}")
def read_recomendacion_juego(id_juego_conocido: int):
    
    
    df = pd.read_parquet("steam_games_clean.parquet")

    # Obtener el género del juego conocido
    genero_a_comparar = df[df['item_id'] == id_juego_conocido]['genres'].values[0]

    # Filtrar el DataFrame para juegos con el mismo género (excluyendo el juego conocido)
    df_filtrado = df[df['item_id'] != id_juego_conocido]

    # Inicializar el vectorizador TF-IDF y aplicarlo a los géneros de los juegos
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtrado['genres'])

    # Vectorizar el género conocido
    vector_genero_conocido = tfidf_vectorizer.transform([genero_a_comparar])

    # Calcular la similitud del coseno entre el género conocido y los géneros de los demás juegos
    similarity_scores = cosine_similarity(vector_genero_conocido, tfidf_matrix)

    # Obtener los índices de los 5 juegos más similares
    indices_juegos_similares = similarity_scores.argsort()[0][-5:][::-1]

    # Obtener los nombres de los 5 juegos más similares en el formato deseado
    juegos_similares = [{"juego": df_filtrado.iloc[idx]['app_name']} for idx in indices_juegos_similares]

    return juegos_similares