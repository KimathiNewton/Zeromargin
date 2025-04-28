# Product Review Analysis Dashboard

A Streamlit web application for comprehensively analyzing customer product reviews using semantic search, sentiment analysis, and large language model (LLM) insights.

## Features

- **Overview**: Key metrics and visualizations of total reviews, categories, average rating, and sentiment distribution.
- **Document Search**: Semantic search powered by Chroma and OpenAI embeddings to find conceptually similar reviews.
- **LLM Insights**: Generate category‑level summaries (performance, praised features, issues, suggestions) using GPT-3.5.
- **Sentiment Analysis**: Real-time text sentiment classification (`positive`, `neutral`, `negative`) and pre‑computed visualizations.
- **Q&A System**: Answer arbitrary product questions by grounding GPT-3.5 responses in retrieved reviews.

## Project Structure

```
├── app.py               # Main Streamlit application
├── utils.py             # Helper functions: retrieval, LLM prompts, visualizations
├── data/                # Data files
│   ├── data_cleaned.csv # Cleaned reviews dataset
│   └── tagged_reviews.txt # Tagged text for Chroma indexing
├── chroma_db/           # Local Chroma vector store (auto‑generated)
├── requirements.txt     # Python package dependencies
└── README.md            # This file
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   - Create a `.env` file in the project root:
     ```ini
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Data Preparation**
   - Ensure `data_cleaned.csv` and `tagged_reviews.txt` are placed in the `data/` directory.
   - On first run, the Chroma vector store will be created under `./chroma_db`.

## Running the App

```bash
streamlit run app.py
```

This will launch the dashboard in your default browser at `http://localhost:8501`.

## Usage

- **Overview**: View summary metrics and interactive Plotly charts.
- **Document Search**: Enter a query (e.g., "battery life") and retrieve semantically similar reviews.
- **LLM Insights**: Select a category, click "Generate Insights", and review the GPT‑3.5 summary.
- **Sentiment Analysis**: Paste any text to analyze sentiment; explore category/time/feature visual tabs.
- **Q&A System**: Ask a question (e.g., "Which product has the best fit?"), view the AI answer and underlying reviews.

