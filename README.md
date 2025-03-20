# Machine Learning Concept Map Generator

This project develops a machine learning model to automatically generate concept maps from textbooks, specifically using "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition" by Aurélien Géron as the data source.

## Project Overview

The model extracts key concepts and their relationships from textbook content using a combination of supervised and unsupervised machine learning techniques, including:

- **Concept Extraction**: Using TF-IDF, TextRank, YAKE, KeyBERT, and SciBERT
- **Relationship Extraction**: Using co-occurrence analysis, dependency parsing, and BERT-based classification
- **Concept Map Construction**: Using NetworkX and Neo4j for graph structure and visualization

## Project Timeline

- **3/2/2025**: Project Proposal Submission
- **3/3 - 3/9/2025**: GitHub setup, initial keyword mapping
- **3/10 - 3/16/2025**: Text preprocessing, initial supervised model implementation
- **3/17 - 3/23/2025**: Model fine-tuning, relationship extraction, Midterm Report
- **3/24 - 3/30/2025**: Model optimization, relationship classification
- **3/31 - 4/6/2025**: Concept map prototype development
- **4/7 - 4/13/2025**: Refinement and validation
- **4/14 - 4/20/2025**: Presentation preparation
- **4/21 - 4/27/2025**: Final report preparation
- **5/4/2025**: Final Report Submission

## Repository Structure

```
ml-concept-map/
├── data/               # Data storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
├── evaluation/         # Evaluation results
├── concept_map/        # Final outputs
└── docs/               # Documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ml-concept-map.git
cd ml-concept-map

# Install dependencies
pip install -r requirements.txt
```

## Team

Daylin Hart Juston Bryant Ashley Archibald

## License

MIT
