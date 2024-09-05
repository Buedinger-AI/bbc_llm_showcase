# app.py
import streamlit as st
import pandas as pd
from src.chunking.chunking import (
    chunk_by_recursive_character,
    chunk_by_character,
    chunk_by_token
)
# File path to your articles CSV
file_path = 'data/articles.csv'
filtered_articles_path = 'data/filtered_articles.csv'  # Path to the filtered articles

# Sidebar configuration
st.sidebar.header("Pipeline Steps")
steps = [
    "1. View Documents",
    "2. Chunking / Parsing Techniques",
    "3. Embedding Models for Vectorization",
    "4. Vector Databases (VectorDBs)",
    "5. Prompt Templates for Query Transformation",
    "6. Retrieval Methods",
    "7. Model Evaluation and Feedback",
    "8. Performance Metrics and Comparisons",
    "9. Error Handling and Explainability"
]
selected_step = st.sidebar.radio("Choose Step:", steps)

# Main area - layout for each step
if selected_step == "1. View Documents":
    st.header("View Documents (News Articles)")
    st.write("Displaying the first 20 filtered articles related to 'Olympia' or 'US Wahlkampf'.")

    # Load the filtered articles into a DataFrame
    try:
        filtered_articles = pd.read_csv(filtered_articles_path)
        top_articles = filtered_articles.head(20).drop(columns=['Unnamed: 0', 'relevant',"section"])  # Limit to the first 20 articles
    except FileNotFoundError:
        st.error(f"File not found: {filtered_articles_path}")
        st.stop()

    # Display articles in a table format
    st.dataframe(top_articles)

    # Optionally, display individual articles with more context
    for index, article in top_articles.iterrows():
        st.subheader(f"Article {index + 1}: {article['headline']}")
        st.write(f"**Published Date:** {article.get('publication_date', 'N/A')}")
        st.write(f"**Link:** {article.get('url', 'N/A')}")
        st.write(f"**Content:** {article['content'][:300]}...")  # Show a snippet of the content
        st.write("---")


elif selected_step == "2. Chunking / Parsing Techniques":
    st.header("Chunking / Parsing Techniques")
    st.write("Explore different chunking and parsing methods using LangChain.")

    # Load the filtered articles
    try:
        filtered_articles = pd.read_csv(filtered_articles_path)
        top_articles = filtered_articles.head(1)  # Use first article for demonstration
        article_text = top_articles.iloc[0]['content']
    except FileNotFoundError:
        st.error(f"File not found: {filtered_articles_path}")
        st.stop()

    # Select chunking method
    chunking_method = st.selectbox(
        "Choose a chunking method:",
        ["Recursive Character-Based", "Simple Character-Based", "Token-Based"]
    )
    
    # Display method explanation
    if chunking_method == "Recursive Character-Based":
        st.markdown("**Recursive Character-Based Chunking**")
        st.write(
            "This method splits the text recursively by different levels (paragraphs, sentences, words) "
            "to create chunks of a defined size. It maintains context better than simple splitting methods "
            "and is ideal for large, structured texts.\n\n"
            "**Advantages**: Preserves context, flexible chunk sizes.\n"
            "**Disadvantages**: Slightly more computationally intensive."
        )
    elif chunking_method == "Simple Character-Based":
        st.markdown("**Simple Character-Based Chunking**")
        st.write(
            "This method splits the text into chunks based on a fixed number of characters. "
            "It's simple and fast, but doesn't consider sentence or paragraph boundaries.\n\n"
            "**Advantages**: Easy to implement, fast processing.\n"
            "**Disadvantages**: May split sentences or words awkwardly, losing context."
        )
    else:
        st.markdown("**Token-Based Chunking**")
        st.write(
            "This method splits the text based on the number of tokens (words or subwords). "
            "It's particularly useful for NLP tasks where token limits matter, such as with language models.\n\n"
            "**Advantages**: Maintains coherence better than character-based splitting, aligns with model token limits.\n"
            "**Disadvantages**: Can still split semantic units if not carefully set."
        )
    
    # Parameters for chunking
    chunk_size = st.slider("Select chunk size:", min_value=100, max_value=2000, value=1000)
    chunk_overlap = st.slider("Select chunk overlap:", min_value=0, max_value=500, value=200)

    # Apply the selected chunking method with overlap
    if chunking_method == "Recursive Character-Based":
        chunks = chunk_by_recursive_character(article_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunking_method == "Simple Character-Based":
        chunks = chunk_by_character(article_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunks = chunk_by_token(article_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Display chunks
    st.write(f"Displaying chunks using {chunking_method} method:")
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(chunk)
        st.write("---")

    st.info(
        "In addition to the chunking methods demonstrated here, LangChain also supports other types "
        "of chunking, such as code, HTML, or Markdown. These specialized chunking methods are designed "
        "to handle structured content types and maintain the integrity of the original formatting."
    )   


elif selected_step == "3. Embedding Models for Vectorization":
    st.header("Embedding Models for Vectorization")
    st.write("Explore different embedding models and their characteristics.")

    # Select embedding model
    embedding_model = st.selectbox(
        "Choose an embedding model:",
        ["Word2Vec", "GloVe", "BERT", "Sentence Transformers (SBERT)", "OpenAI GPT Embeddings"]
    )

    # Display explanations for each embedding model
    if embedding_model == "Word2Vec":
        st.markdown("**Word2Vec**")
        st.write(
            "Word2Vec is one of the earliest models that converts words into vectors based on their "
            "context in sentences. It uses either Skip-gram or CBOW (Continuous Bag of Words) approaches "
            "to learn word representations.\n\n"
            "**Advantages**: Fast and resource-efficient, good for smaller datasets.\n"
            "**Disadvantages**: Context-independent, loses meaning of ambiguities and ignores sentence structure."
        )
    elif embedding_model == "GloVe":
        st.markdown("**GloVe (Global Vectors for Word Representation)**")
        st.write(
            "GloVe uses a global statistical approach to capture the relationships between words across "
            "a large corpus, producing word vectors that reflect these relationships.\n\n"
            "**Advantages**: Considers global contexts, handles rare words better than Word2Vec.\n"
            "**Disadvantages**: Still context-independent and static."
        )
    elif embedding_model == "BERT":
        st.markdown("**BERT (Bidirectional Encoder Representations from Transformers)**")
        st.write(
            "BERT is a transformer-based model that processes words in their bidirectional context, "
            "allowing it to deeply understand the meaning of words in a sentence.\n\n"
            "**Advantages**: Context-dependent, understands the relationship between words in a sentence.\n"
            "**Disadvantages**: Computationally intensive, slower in processing large datasets."
        )
    elif embedding_model == "Sentence Transformers (SBERT)":
        st.markdown("**Sentence Transformers (e.g., SBERT)**")
        st.write(
            "Sentence Transformers extend BERT to create embeddings for sentences and documents, optimized "
            "for semantic similarity tasks and retrieval applications.\n\n"
            "**Advantages**: Faster similarity calculations, excellent for retrieval tasks.\n"
            "**Disadvantages**: May be less precise than specialized models in certain contexts."
        )
    else:
        st.markdown("**OpenAI GPT Embeddings (GPT-3, GPT-4)**")
        st.write(
            "OpenAI's GPT models use transformer architectures to encode words, sentences, and documents "
            "into high-dimensional spaces, capturing complex linguistic structures and contexts.\n\n"
            "**Advantages**: Highly powerful, understands complex linguistic structures.\n"
            "**Disadvantages**: Very high computational and storage costs, expensive to use."
        )

    # Additional note on using embeddings
    st.info(
        "Embedding models are critical for transforming text into numerical formats that capture meaning. "
        "Choosing the right embedding model depends on the specific application, data size, and performance requirements."
    )

elif selected_step == "4. Vector Databases (VectorDBs)":
    st.header("Vector Databases (VectorDBs)")
    st.write("Showcase various vector databases and their performance.")
    # Placeholder for VectorDB details and comparisons
    st.info("Vector database performance comparisons will be displayed here.")

elif selected_step == "5. Prompt Templates for Query Transformation":
    st.header("Prompt Templates for Query Transformation")
    st.write("Demonstrate different prompt templates and their effects on query transformation.")
    # Placeholder for prompt template comparisons
    st.info("Prompt template impacts on query generation will be shown here.")

elif selected_step == "6. Retrieval Methods":
    st.header("Retrieval Methods")
    st.write("Present different retrieval strategies and their effectiveness.")
    # Placeholder for retrieval comparisons
    st.info("Retrieval strategies and results will be displayed here.")

elif selected_step == "7. Model Evaluation and Feedback":
    st.header("Model Evaluation and Feedback")
    st.write("Allow users to evaluate generated answers and provide feedback.")
    # Placeholder for evaluation and feedback mechanism
    generated_answer = "Sample generated answer."
    st.write(generated_answer)
    feedback = st.radio("How would you rate this answer?", ["Good", "Average", "Poor"])
    comment = st.text_area("Any suggestions for improvement?")

elif selected_step == "8. Performance Metrics and Comparisons":
    st.header("Performance Metrics and Comparisons")
    st.write("Compare the performance of different configurations across the pipeline.")
    # Placeholder for performance metrics and charts
    st.info("Performance metrics and comparisons will be visualized here.")

elif selected_step == "9. Error Handling and Explainability":
    st.header("Error Handling and Explainability")
    st.write("Explain common errors and provide insights into the pipeline.")
    # Placeholder for error handling and explainability
    st.info("Error handling insights and explanations will be shown here.")

# Footer or additional notes
st.markdown("---")
st.write("This app is a showcase of the different stages involved in a RAG pipeline for Q&A using BBC News data.")

