# src/utils/data_preprocessor.py
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def filter_articles_with_llm(file_path: str, output_path: str):
    """
    Filters articles using an LLM to identify those related to 'Olympia' or 'US Wahlkampf'.

    Args:
        file_path (str): Path to the input CSV file containing articles.
        output_path (str): Path to save the filtered articles CSV.

    Returns:
        None
    """
    # Load the articles from the CSV file
    try:
        articles = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception(f"File not found at {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")

    # Initialize the LLM with the specified model
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from the environment
    if not openai_api_key:
        raise Exception("OpenAI API key is not set. Please check your .env file.")

    # Initialize the ChatOpenAI with the custom model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.0,
        model_name="gpt-4o-mini"  # Using your specified model
    )

    def is_relevant(article):
        """
        Uses an LLM to determine if an article is relevant to 'Olympia' or 'US Wahlkampf'.
        
        Args:
            article (str): The content of the article to check.

        Returns:
            bool: True if the article is relevant, False otherwise.
        """
        # Prepare the chat format
        messages = [
            {"role": "system", "content": "You are an assistant that identifies relevant articles."},
            {"role": "user", "content": f"Is the following article related to the Olympics ('Olympia') or US Presidential Election ('US Wahlkampf')? Please respond with 'Yes' or 'No'.\n\nArticle: {article}"}
        ]
        
        # Use invoke to properly call the LLM
        response = llm.invoke(messages)
        
        # Access the content of the response correctly
        if hasattr(response, 'content'):
            return "yes" in response.content.lower()
        else:
            return False

    # Apply the filtering using the LLM
    articles['relevant'] = articles['content'].apply(is_relevant)
    filtered_articles = articles[articles['relevant']]

    # Save the filtered articles
    filtered_articles.to_csv(output_path, index=False)
    print(f"Filtered articles saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_path = 'data/articles.csv'
    output_path = 'data/filtered_articles.csv'
    filter_articles_with_llm(input_path, output_path)
