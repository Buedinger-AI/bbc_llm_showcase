import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone(api_key, environment, index_name, dimension=384):
    """
    Initialisiert Pinecone und erstellt bei Bedarf einen neuen Index.
    
    Args:
    - api_key (str): Der API-Schlüssel für Pinecone.
    - environment (str): Die Umgebung, z.B. 'us-west1-gcp'.
    - index_name (str): Der Name des zu verwendenden oder zu erstellenden Index.
    - dimension (int): Die Dimension der Vektoren, die gespeichert werden.
    
    Returns:
    - index (Index): Der Pinecone-Index, der verwendet wird.
    """
    # Erstelle eine Instanz der Pinecone-Klasse
    pc = Pinecone(api_key=api_key)
    
    # Extrahiere Cloud und Region aus der environment-Variable
    
    region = "us-east-1"  # z.B. 'us-west1'
    cloud = "aws"   # z.B. 'gcp'

    # Überprüfe, ob der Index existiert, und erstelle ihn bei Bedarf
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',  # Du kannst die Metrik anpassen, falls nötig
            spec=ServerlessSpec(
                cloud=cloud,     # 'gcp', 'aws', 'azure'
                region=region    # z.B. 'us-west1'
            )
        )
    
    # Greife auf den Index zu
    index = pc.Index(index_name)
    
    return index

def load_and_preprocess_data(csv_file, model_name='all-MiniLM-L6-v2'):
    """
    Lädt die Daten aus einer CSV-Datei und verarbeitet sie mit einem SentenceTransformer-Modell.
    
    Args:
    - csv_file (str): Pfad zur CSV-Datei.
    - model_name (str): Der Name des SentenceTransformer-Modells.
    
    Returns:
    - df (pd.DataFrame): Das DataFrame mit den ursprünglichen Daten und den neuen Vektor-Embeddings.
    - model (SentenceTransformer): Das geladene SentenceTransformer-Modell.
    """
    df = pd.read_csv(csv_file)
    model = SentenceTransformer(model_name)
    df['content_embedding'] = df['content'].apply(lambda x: model.encode(x).tolist())
    return df, model

def upsert_data_to_pinecone(index, df):
    """
    Fügt die verarbeiteten Daten in den Pinecone-Index ein oder aktualisiert sie.
    
    Args:
    - index (Index): Der Pinecone-Index.
    - df (pd.DataFrame): Das DataFrame mit den Daten und Vektor-Embeddings.
    """
    for idx, row in df.iterrows():
        index.upsert(vectors=[(str(idx), row['content_embedding'], {"headline": row['headline'], "url": row['url']})])
