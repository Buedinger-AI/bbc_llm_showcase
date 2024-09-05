import streamlit as st
from utils.pinecone_util import initialize_pinecone, load_and_preprocess_data, upsert_data_to_pinecone
from dotenv import load_dotenv
import os

# Laden der Umgebungsvariablen aus der .env Datei
load_dotenv()

# API-Schlüssel aus der .env Datei laden
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = "us-east-1"
INDEX_NAME = 'news-articles-index'

# Sicherstellen, dass alle notwendigen API-Schlüssel geladen wurden
if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Bitte stelle sicher, dass alle API-Schlüssel und die Umgebung korrekt in der .env Datei gesetzt sind.")
    st.stop()

# Setup Pinecone
index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME)

# Datenverarbeitung und Pinecone-Indexierung (Einmalige Ausführung)
csv_file = 'articles.csv'
df, model = load_and_preprocess_data(csv_file, model_name='all-MiniLM-L6-v2')
upsert_data_to_pinecone(index, df)

st.success("Daten wurden erfolgreich in Pinecone hochgeladen und indiziert!")
