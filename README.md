
This repository,  contains three main projects focused on Machine Learning, Natural Language Processing, and MLOps best practices. Each part is presented in its own folder:

- **Part 1**: Machine Learning Challenge (retail sales forecasting & customer segmentation)  
- **Part 2**: LLM & Vector Database Challenge (product review analysis & retrieval)  
- **Part 3**: MLOps Architecture Design (deployment & governance of AI models)

---

## Part 1: Machine Learning Challenge
Location: [Part1 Folder](./Part1/)


Using a multi-store retail dataset (daily sales across 10 stores & 5 categories, Jan–Dec 2023), this part covers:

1. **Data Preprocessing**
   - Load & clean the dataset, handling missing values
   - Perform Exploratory Data Analysis (EDA) to uncover trends
   - Engineer features to improve model performance
   - Encode, scale, and prepare data for modeling

2. **Sales Forecasting**
   - Build a model to forecast daily total sales per store-category
   - Generate a 14-day forecast for January 2024
   - Evaluate forecasts with metrics (e.g., RMSE, MAE)
   - Analyze feature importance for prediction insights

3. **Customer Segmentation**
   - Cluster stores into meaningful segments using features
   - Provide actionable insights per segment
   - Suggest tailored strategies for each segment

---

## Part 2: LLM & Vector Database Challenge
Location: [Part2 Folder](./Part2/)


Working with electronic product reviews:

1. **Document Processing Pipeline**
   - Parse and extract key fields from reviews
   - Generate embeddings (OpenAI, Hugging Face, or similar)
   - Store & retrieve reviews by semantic similarity

2. **LLM Application**
   - Summarize product performance by category
   - Build a Q&A system over the review dataset
   - Identify common issues and praised features

3. **Sentiment Analysis & Classification**
   - Implement automated sentiment classifier
   - Compare predictions vs. provided labels
   - Create a dashboard of sentiment trends over time & by category

---

## Part 3: MLOps Architecture Design
Location: [Part3 Folder](./Part3/)

Design of a production-grade system for three AI models:

1. **MLOps Architecture**
   - Defined end-to-end deployment, monitoring & maintenance
   - Includes scalability, reliability, and security considerations
   - An architecture diagram

2. **Data & Model Versioning**
   - Strategy for versioning datasets and model artifacts
   - Plan for updates and rollbacks
   - Track lineage and ensure reproducibility

3. **Monitoring & Maintenance**
   - Key performance and drift metrics
   - Alerting system for degradation
   - Retraining triggers and processes

4. **Documentation & Governance**
   - Document model behavior and limitations
   - Governance framework for approvals and updates
   - Compliance with relevant standards and regulations

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KimathiNewton/Zeromargin.git
   ```
2. **Navigate to the part folder**:
   ```bash
   cd Zeromargin/Part1_RetailSales   # or Part2_ReviewAnalytics, Part3_MLOpsDesign
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Follow the README in each folder** for detailed instructions.



