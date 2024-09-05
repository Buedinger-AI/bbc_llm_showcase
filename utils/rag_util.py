import openai


def set_openai_api_key(api_key):
    openai.api_key = api_key

def retrieve_articles(query, model, index, top_k=3):
    query_embedding = model.encode(query).tolist()
    result = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return result['matches']

def generate_response(query, articles):
    # Kombinieren der abgerufenen Artikel
    context = "\n\n".join([f"{article['metadata']['headline']}: {article['metadata']['url']}" for article in articles])
    prompt = f"Hier ist eine Zusammenfassung der relevantesten Informationen:\n\n{context}\n\nFrage: {query}\nAntwort:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # GPT-4 Engine
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()