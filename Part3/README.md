
## Overview
This folder contains the finalized **MLOps Design Document** and an **enhanced architecture diagram** for Zero Margin Limited.  
The architecture outlines a scalable, secure, and governed end-to-end machine learning operational pipeline, hosted on **Google Cloud Platform (GCP)**.

## Contents
- `MLOps_Design_Document.pdf` — Full detailed document describing the architecture, workflows, and governance mechanisms.
- `MLOps_Architecture_Diagram.png` — Updated system architecture diagram showcasing CI/CD, Feature Store, versioning, retraining, and governance workflows.

## Architecture Highlights
- **Secure Boundary:** Data pipelines, model training, feature stores, and serving clusters are protected within VPCs, IAM policies, and Secret Manager.
- **Feature Store:** Central repository for engineered features, supporting both training and online prediction.
- **CI/CD Integration:** Automated pipelines using Cloud Build and Vertex AI ensure consistent versioning of data, models, and artifacts.
- **Governance Framework:** Formal approval gates, Model Cards, DPIA documentation, and GDPR compliance integrated into the deployment lifecycle.
- **Monitoring & Retraining:** Custom metrics monitored with Prometheus and Grafana; automated retraining triggered on drift detection or performance degradation.

## Diagram Improvements
- Added missing components: **CI/CD pipeline, Feature Store**, and **Governance policies**.
- Corrected typos and standardized terminology.
- Explicitly depicted **versioning**, **retraining loops**, and **governance processes** to align the visual with the written design.

## Getting Started
1. Review the `MLOps_Design_Document.pdf` for the detailed description of each module.
2. Refer to `MLOps_Architecture_Diagram.png` for a quick system overview.
3. Adapt and customize the architecture to your organization's needs and GCP setup.

