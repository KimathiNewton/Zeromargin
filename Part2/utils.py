## Importations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import pipeline
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sentiment Analysis Function
def get_sentiment(text):
    """Get sentiment prediction for a text"""
    try:
        # Load model
        model_name = "j-hartmann/sentiment-roberta-large-english-3-classes"
        classifier = pipeline("text-classification", model=model_name)
        # Get top prediction
        result = classifier(text)[0]
        return result["label"]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Error"

# Vector Search Function
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Global Chroma DB client (initialized on first use)
_db: Chroma = None

def _get_db() -> Chroma:
    """Initialize or return a cached local Chroma vector store."""
    global _db
    if _db is None:
        # 1. Load and split source documents
        raw_docs = TextLoader("tagged_reviews.txt").load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
        docs = splitter.split_documents(raw_docs)

        # 2. Create a local, on-disk Chroma store
        _db = Chroma.from_documents(
            docs,
            embedding=OpenAIEmbeddings(),
            persist_directory="chroma_db",    # stores vector data under ./chroma_db
            collection_name="reviews"
        )
    return _db


def retrieve_semantic_reviews(
    query: str,
    df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Retrieve the top_k semantically similar reviews from df for the given query."""
    # 1. Query the local Chroma store
    db = _get_db()
    recs = db.similarity_search(query, k=top_k)

    # 2. Extract review IDs (first token of each chunk)
    review_ids = []
    for r in recs:
        tok = r.page_content.strip('"').split()[0]
        try:
            review_ids.append(int(tok))
        except ValueError:
            review_ids.append(tok)

    # 3. Filter the DataFrame and preserve Chroma's ranking
    filtered = (
        df
        .set_index("review_id")
        .loc[review_ids]
        .reset_index()
    )
    return filtered


# LLM Application Functions
def generate_category_summary(df, category):
    """Generate a summary for a specific product category"""
    # 1. Filter data for the category
    category_data = df[df['category'] == category]
    
    # 1a. If there are no reviews, bail out immediately
    if category_data.empty:
        return f"No reviews found for category '{category}'."

    # 2. Sample up to 10 reviews
    sample_reviews = category_data.sample(min(10, len(category_data)))

    # 3. Build the prompt
    reviews_text = "\n\n".join(
        f"Rating: {row['rating']}, Review: {row['review_text']}"
        for _, row in sample_reviews.iterrows()
    )
    prompt = (
        f"Based on the following customer reviews for {category} products:\n\n"
        f"{reviews_text}\n\n"
        "Please provide:\n"
        "1. A concise summary of overall product performance\n"
        "2. Top praised features\n"
        "3. Common issues mentioned\n"
        "4. Suggestions for improvement\n"
    )

    # 4. Call the LLM
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error generating summary for {category}: {e}"

from typing import Callable

def product_qa(
    question: str,
    df: pd.DataFrame,
    retrieve_fn: Callable[[str], pd.DataFrame],
    top_k: int = 5,
) -> str:
    """Answer questions about products based on relevant reviews."""
    # 1. Retrieve candidate reviews
    relevant = retrieve_fn(question)
    # 1a. If there are no relevant reviews, bail out early
    if relevant.empty:
        return "I couldn't find any reviews relevant to your question."

    # 2. Keep only the top_k most relevant (preserving order)
    top_reviews = relevant.head(top_k)

    # 3. Build a concise prompt from the top reviews
    reviews_text = "\n\n".join(
        f"Product: {row['product']}, Rating: {row['rating']}, Review: {row['review_text']}"
        for _, row in top_reviews.iterrows()
    )

    # 4. Construct a system + user prompt to ground the LLM
    system_message = (
        "You are a precise product review assistant. "
        "Answer the user's question strictly based on the provided reviews."
    )
    user_prompt = (
        f"Here are some customer reviews about products:\n\n{reviews_text}\n\n"
        f"Question: {question}\n\nAnswer:" 
    )

    # 5. Call the LLM and return its answer
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in Q&A system: {e}")
        return f"Sorry, I couldn't answer your question due to an error: {e}"

# Visualization Functions
def create_sentiment_visualizations(df):
    """Create various sentiment analysis visualizations"""
    # Make sure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Add month and year columns for easier grouping
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # 1. Sentiment distribution by category
    plt.figure(figsize=(12, 8))
    sentiment_by_category = pd.crosstab(df['category'], df['sentiment'])
    sentiment_by_category_pct = sentiment_by_category.div(sentiment_by_category.sum(axis=1), axis=0)
    sentiment_by_category_pct.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Sentiment Distribution by Product Category')
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_by_category.png')
    
    # 2. Sentiment trends over time
    plt.figure(figsize=(14, 8))
    monthly_sentiment = df.groupby([df['year'], df['month'], 'sentiment']).size().unstack()
    monthly_sentiment.plot(kind='line', marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Year, Month')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_trends.png')
    
    # 3. Correlation between rating and sentiment
    plt.figure(figsize=(10, 6))
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'])
    sns.heatmap(rating_sentiment, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Correlation between Rating and Sentiment')
    plt.tight_layout()
    plt.savefig('rating_sentiment_correlation.png')
    
    # 4. Sentiment by feature mentioned
    plt.figure(figsize=(14, 8))
    top_features = df['feature_mentioned'].value_counts().head(10).index
    feature_sentiment = pd.crosstab(
        df[df['feature_mentioned'].isin(top_features)]['feature_mentioned'], 
        df[df['feature_mentioned'].isin(top_features)]['sentiment']
    )
    feature_sentiment_pct = feature_sentiment.div(feature_sentiment.sum(axis=1), axis=0)
    feature_sentiment_pct.plot(kind='barh', stacked=True, colormap='viridis')
    plt.title('Sentiment by Top Feature Mentioned')
    plt.xlabel('Percentage')
    plt.tight_layout()
    plt.savefig('feature_sentiment.png')
    
    return {
        'sentiment_by_category': 'sentiment_by_category.png',
        'sentiment_trends': 'sentiment_trends.png',
        'rating_sentiment_correlation': 'rating_sentiment_correlation.png',
        'feature_sentiment': 'feature_sentiment.png'
    }