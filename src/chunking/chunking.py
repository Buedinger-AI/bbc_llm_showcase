# src/chunking/chunking.py
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

def chunk_by_recursive_character(text, chunk_size=1000, chunk_overlap=200):
    """
    Uses RecursiveCharacterTextSplitter to chunk text with overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_text(text)

def chunk_by_character(text, chunk_size=1000, chunk_overlap=200):
    """
    Uses CharacterTextSplitter to chunk text with overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = current_pos + chunk_size
        chunk = text[current_pos:end_pos]
        chunks.append(chunk)
        current_pos += chunk_size - chunk_overlap  # Adjust position for overlap
    return chunks

def chunk_by_token(text, chunk_size=200, chunk_overlap=50):
    """
    Uses TokenTextSplitter to chunk text with overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)
