import streamlit as st
from utils.pinecone_util import initialize_pinecone, load_and_preprocess_data, upsert_data_to_pinecone
from utils.rag_util import set_openai_api_key, retrieve_articles, generate_response
from dotenv import load_dotenv
import os

# Laden der Umgebungsvariablen aus der .env Datei
load_dotenv()

# API-Schlüssel aus der .env Datei laden
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = "us-east-1"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = 'news-index'

# Sicherstellen, dass alle notwendigen API-Schlüssel geladen wurden
if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_ENV:
    st.error("Bitte stelle sicher, dass alle API-Schlüssel und die Umgebung korrekt in der .env Datei gesetzt sind.")
    st.stop()

# Setup Pinecone und OpenAI
index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME)
set_openai_api_key(OPENAI_API_KEY)

# Datenverarbeitung und Pinecone-Indexierung (Einmalige Ausführung)
csv_file = 'articles.csv'
df, model = load_and_preprocess_data(csv_file, model_name='text-embedding-ada-002')
upsert_data_to_pinecone(index, df)

st.success("Daten wurden erfolgreich in Pinecone hochgeladen und indiziert!")

# Einführung und App-Titel
st.title("AI News Insight Showcase")
st.write("""
Willkommen zur **AI News Insight Showcase** App! Diese Anwendung demonstriert moderne Generative AI-Techniken und wie sie auf Nachrichteninhalte angewendet werden können.
Nutzen Sie die Navigationsleiste auf der linken Seite, um verschiedene Funktionen und Techniken zu erkunden.
""")

# Sidebar für die Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Wähle einen Abschnitt", [
    "Einführung",
    "Retrieval-Augmented Generation (RAG)",
    "In-Context Learning",
    "Prompt Engineering",
    "Chain-of-Thought Reasoning",
    "Dynamic Few-Shot Learning",
    "Long-Context Models",
    "Multi-Modal Generative Models",
    "Explainability in Generative AI",
    "Contextual and Temporal Awareness",
    "Interactive and Conversational AI"
])

# Content-Bereich je nach Auswahl in der Sidebar
if selection == "Einführung":
    st.header("Einführung")
    st.write("""
    Diese App zeigt eine Vielzahl moderner Techniken im Bereich der Generativen Künstlichen Intelligenz. 
    Jeder Abschnitt bietet eine praktische Demonstration einer spezifischen Technik, die in der modernen 
    NLP (Natural Language Processing) und AI (Artificial Intelligence) eingesetzt wird.
    Nutzen Sie die Sidebar, um durch die verschiedenen Bereiche der App zu navigieren.
    """)
elif selection == "Retrieval-Augmented Generation (RAG)":
    st.header("Retrieval-Augmented Generation (RAG)")
    st.write("""
    **Retrieval-Augmented Generation (RAG)** kombiniert das Abrufen von Informationen mit generativen Modellen, 
    um präzise und kontextuell relevante Antworten zu generieren.
    In diesem Abschnitt zeigen wir, wie ein Modell Inhalte aus bestehenden Nachrichtenartikeln abruft und 
    diese verwendet, um Antworten auf benutzergenerierte Fragen zu erstellen.
    """)

# Weitere Abschnitte für jede Technik
elif selection == "In-Context Learning":
    st.header("In-Context Learning")
    st.write("""
    **In-Context Learning** ermöglicht es dem Modell, durch Bereitstellung von Beispielen im Kontext 
    bessere und spezifischere Ergebnisse zu erzielen.
    In diesem Abschnitt werden wir Beispiele für unterschiedliche Eingabeaufforderungen testen und 
    sehen, wie das Modell darauf reagiert.
    """)

elif selection == "Prompt Engineering":
    st.header("Prompt Engineering")
    st.write("""
    **Prompt Engineering** bezieht sich auf die Optimierung von Texteingaben (Prompts), um die 
    bestmöglichen Ergebnisse von generativen Modellen zu erhalten. 
    Hier können Sie verschiedene Prompts ausprobieren und die Auswirkungen auf den generierten 
    Text beobachten.
    """)

elif selection == "Chain-of-Thought Reasoning":
    st.header("Chain-of-Thought Reasoning")
    st.write("""
    **Chain-of-Thought Reasoning** ermöglicht es dem Modell, komplexe Aufgaben zu lösen, 
    indem es seine Denkprozesse explizit darstellt.
    Hier sehen Sie, wie das Modell seine Schritte aufzeigt, um zu einer Antwort zu kommen.
    """)

elif selection == "Dynamic Few-Shot Learning":
    st.header("Dynamic Few-Shot Learning")
    st.write("""
    **Dynamic Few-Shot Learning** zeigt, wie ein Modell dynamisch wenige Beispiele auswählt, 
    um neue Aufgaben zu lösen.
    In diesem Abschnitt können Sie sehen, wie das Modell diese Beispiele verwendet, um 
    sich an neue Aufgaben anzupassen.
    """)

elif selection == "Long-Context Models":
    st.header("Long-Context Models")
    st.write("""
    **Long-Context Models** sind in der Lage, lange Textsequenzen zu verarbeiten und den 
    Zusammenhang über längere Dokumente oder Konversationen hinweg zu bewahren.
    Hier können Sie sehen, wie das Modell konsistente Inhalte über längere Texte hinweg generiert.
    """)

elif selection == "Multi-Modal Generative Models":
    st.header("Multi-Modal Generative Models")
    st.write("""
    **Multi-Modal Generative Models** können Text- und Bilddaten kombinieren, um umfassendere Inhalte zu erstellen.
    In diesem Abschnitt können Sie eine Textbeschreibung eingeben und sehen, wie das Modell dazu passende 
    Bilder generiert.
    """)

elif selection == "Explainability in Generative AI":
    st.header("Explainability in Generative AI")
    st.write("""
    **Explainability in Generative AI** zielt darauf ab, die Entscheidungen von generativen Modellen transparent 
    und nachvollziehbar zu machen.
    Hier können Sie generierte Inhalte analysieren und verstehen, wie und warum diese erstellt wurden.
    """)

elif selection == "Contextual and Temporal Awareness":
    st.header("Contextual and Temporal Awareness in Generation")
    st.write("""
    **Contextual and Temporal Awareness** berücksichtigt den zeitlichen und kontextuellen Rahmen, um relevante Inhalte zu generieren.
    In diesem Abschnitt wird gezeigt, wie das Modell den Kontext über Zeiträume hinweg nutzt, um genauere Ergebnisse zu erzielen.
    """)

elif selection == "Interactive and Conversational AI":
    st.header("Interactive and Conversational AI")
    st.write("""
    **Interactive and Conversational AI** ermöglicht es Benutzern, in Echtzeit mit einem Modell zu interagieren und relevante Inhalte zu generieren.
    Hier können Sie eine Unterhaltung mit dem Modell führen und sehen, wie es auf Ihre Anfragen reagiert.
    """)

