import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

# Import your custom functions
# You'll need to create a utils.py file with the functions we developed above
from utils import (generate_category_summary, product_qa, retrieve_semantic_reviews, 
                   get_sentiment, create_sentiment_visualizations)

# Set page config
st.set_page_config(
    page_title="Product Review Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    return pd.read_csv("data_cleaned.csv")

# Load vector database
@st.cache_resource
def load_vectordb():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    
    try:
        raw_documents = TextLoader("tagged_reviews.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(raw_documents)
        
        db = Chroma.from_documents(
            documents,
            embedding=OpenAIEmbeddings(),
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None

# Initialize sentiment classifier
@st.cache_resource
def load_sentiment_classifier():
    try:
        model_name = "j-hartmann/sentiment-roberta-large-english-3-classes"
        return pipeline("text-classification", model=model_name)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

# Main function
def main():
    # App title
    st.markdown("<h1 class='main-header'>Product Review Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Document Search", "LLM Insights", "Sentiment Analysis", "Q&A System"]
    )
    
    # Overview page
    if page == "Overview":
        display_overview(df)
    
    # Document Search page
    elif page == "Document Search":
        display_document_search(df)
    
    # LLM Insights page
    elif page == "LLM Insights":
        display_llm_insights(df)
    
    # Sentiment Analysis page
    elif page == "Sentiment Analysis":
        display_sentiment_analysis(df)
    
    # Q&A System page
    elif page == "Q&A System":
        display_qa_system(df)

def display_overview(df):
    st.markdown("<h2 class='section-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-label'>Total Reviews</div>
                <div class='metric-value'>{len(df)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-label'>Product Categories</div>
                <div class='metric-value'>{df['category'].nunique()}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-label'>Average Rating</div>
                <div class='metric-value'>{df['rating'].mean():.1f}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        positive_pct = (df['sentiment'] == 'positive').mean() * 100
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-label'>Positive Sentiment</div>
                <div class='metric-value'>{positive_pct:.1f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Data sample
    st.markdown("<h3>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(df.head())
    
    # Distribution visualizations
    st.markdown("<h3>Data Distributions</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        fig = px.bar(
            df['category'].value_counts().reset_index(),
            x='category', y='count',
            title='Reviews by Product Category',
            labels={'count': 'Number of Reviews', 'category': 'Category'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rating distribution
        fig = px.histogram(
            df, x='rating',
            title='Rating Distribution',
            labels={'rating': 'Rating', 'count': 'Number of Reviews'},
            nbins=5
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution
    fig = px.pie(
        df, names='sentiment',
        title='Sentiment Distribution',
        color='sentiment',
        color_discrete_map={'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_document_search(df):
    st.markdown("<h2 class='section-header'>Semantic Search</h2>", unsafe_allow_html=True)
    
    # Search interface
    query = st.text_input("Enter your search query:", "battery life")
    top_k = st.slider("Number of results to return:", min_value=1, max_value=20, value=5)
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                # Call your retrieve_semantic_reviews function
                results = retrieve_semantic_reviews(query,df, top_k)
                
                if len(results) > 0:
                    st.success(f"Found {len(results)} relevant reviews!")
                    
                    for i, (_, row) in enumerate(results.iterrows()):
                        st.markdown(f"""
                        <div class='card'>
                            <h4>Review {i+1}: {row['product']}</h4>
                            <p><strong>Category:</strong> {row['category']}</p>
                            <p><strong>Rating:</strong> {'⭐' * int(row['rating'])}</p>
                            <p><strong>Sentiment:</strong> {row['sentiment'].capitalize()}</p>
                            <p><strong>Review:</strong> {row['review_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No relevant reviews found.")
            except Exception as e:
                st.error(f"Error performing search: {e}")
    
    st.markdown("""
    #### How it works
    The semantic search uses document embeddings to find reviews that are conceptually similar to your query, 
    even if they don't contain the exact same words. This allows you to search for concepts and ideas
    rather than just keywords.
    """)

def display_llm_insights(df):
    st.markdown("<h2 class='section-header'>LLM-Generated Insights</h2>", unsafe_allow_html=True)
    
    # Category selection
    categories = ['All Categories'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Select Product Category:", categories)
    
    if st.button("Generate Insights"):
        with st.spinner("Generating insights... This may take a moment."):
            try:
                if selected_category == 'All Categories':
                    # Generate insights for all categories (might be slow)
                    for category in df['category'].unique():
                        st.markdown(f"### {category}")
                        summary = generate_category_summary(df, category)
                        st.markdown(summary)
                        st.divider()
                else:
                    # Generate insights for selected category
                    summary = generate_category_summary(df, selected_category)
                    st.markdown(f"### {selected_category}")
                    st.markdown(summary)
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    
    st.markdown("""
    #### About This Feature
    This feature uses a large language model to analyze the reviews for your selected product category
    and generate insights about overall performance, praised features, and common issues.
    """)

def display_sentiment_analysis(df):
    st.markdown("<h2 class='section-header'>Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    # Real-time sentiment analyzer
    st.markdown("### Try the Sentiment Analyzer")
    test_text = st.text_area("Enter text to analyze:", "This product exceeds my expectations, I love it!")
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            try:
                sentiment = get_sentiment(test_text)
                
                # Display result with appropriate color
                if sentiment.lower() == "positive":
                    st.success(f"Sentiment: {sentiment}")
                elif sentiment.lower() == "negative":
                    st.error(f"Sentiment: {sentiment}")
                else:
                    st.info(f"Sentiment: {sentiment}")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")
    
    # Display pre-generated visualizations
    st.markdown("### Sentiment Analysis Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["By Category", "Over Time", "By Rating", "By Feature"])
    
    with tab1:
        # Sentiment by category visualization
        fig = px.bar(
            pd.crosstab(df['category'], df['sentiment'], normalize='index') * 100,
            title="Sentiment Distribution by Category",
            labels={'value': 'Percentage', 'sentiment': 'Sentiment'},
            color_discrete_map={'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Time series visualization
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = df['date'].dt.strftime('%Y-%m')
        
        sentiment_time = df.groupby(['month_year', 'sentiment']).size().reset_index(name='count')
        pivot_sentiment = sentiment_time.pivot(index='month_year', columns='sentiment', values='count').fillna(0)
        
        fig = px.line(
            pivot_sentiment, 
            title="Sentiment Trends Over Time",
            labels={'value': 'Number of Reviews', 'variable': 'Sentiment'},
            color_discrete_map={'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation between rating and sentiment
        heatmap_data = pd.crosstab(df['rating'], df['sentiment'])
        
        fig = px.imshow(
            heatmap_data,
            title="Rating vs. Sentiment Correlation",
            labels=dict(x="Sentiment", y="Rating", color="Count"),
            color_continuous_scale='viridis',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Top features by sentiment
        top_features = df['feature_mentioned'].value_counts().head(10).index.tolist()
        feature_sentiment = pd.crosstab(
            df[df['feature_mentioned'].isin(top_features)]['feature_mentioned'],
            df[df['feature_mentioned'].isin(top_features)]['sentiment'],
            normalize='index'
        ) * 100
        
        fig = px.bar(
            feature_sentiment,
            title="Sentiment Distribution by Top Features",
            labels={'value': 'Percentage', 'sentiment': 'Sentiment'},
            orientation='h',
            color_discrete_map={'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    st.markdown("### Model Performance")
    
    # Load or calculate performance metrics
    try:
        # You should calculate these metrics beforehand or load them from a file
        accuracy = 0.85  # Replace with actual accuracy
        precision = 0.87  # Replace with actual precision
        recall = 0.83  # Replace with actual recall
        f1 = 0.85  # Replace with actual F1 score
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
        
        with col2:
            st.metric("Precision", f"{precision:.2f}")
        
        with col3:
            st.metric("Recall", f"{recall:.2f}")
        
        with col4:
            st.metric("F1 Score", f"{f1:.2f}")
    
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")

def display_qa_system(df):
    st.markdown("<h2 class='section-header'>Product Q&A System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Ask any question about the products in the dataset, and our AI will answer based on
    the customer reviews.
    
    Example questions:
    - Which smartphone has the best battery life?
    - What are common complaints about the laptops?
    - Which smart home device is most reliable?
    - Are the wireless earbuds comfortable to wear?
    """)
    
    question = st.text_input("Enter your question:", "Which product has the best battery life?")
    
    if st.button("Get Answer"):
        with st.spinner("Finding answer... This may take a moment."):
            try:
                # Call your product_qa function
                answer = product_qa(question, df, lambda q: retrieve_semantic_reviews(q,df, top_k=10))
                
                st.markdown(f"""
                <div class='card'>
                    <h3>Answer:</h3>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show relevant reviews used for the answer
                with st.expander("See relevant reviews used for this answer"):
                    relevant_reviews = retrieve_semantic_reviews(question, df, top_k=5)
                    for i, (_, row) in enumerate(relevant_reviews.iterrows()):
                        st.markdown(f"""
                        <div style='margin-bottom: 10px; padding: 10px; border-left: 3px solid #2563EB;'>
                            <p><strong>Product:</strong> {row['product']}</p>
                            <p><strong>Rating:</strong> {'⭐' * int(row['rating'])}</p>
                            <p><strong>Review:</strong> {row['review_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error getting answer: {e}")
    
    st.markdown("""
    #### How it works
    The Q&A system combines semantic search to find relevant reviews and a large language model
    to generate a comprehensive answer based on these reviews. The system only uses information
    contained in the reviews to formulate its answers.
    """)

if __name__ == "__main__":
    main()