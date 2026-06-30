
<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&amp;color=0:0F172A,35:2563EB,70:14B8A6,100:22C55E&amp;height=220&amp;section=header&amp;text=Advanced%20Information%20Retrieval&amp;fontSize=34&amp;fontColor=ffffff&amp;fontAlignY=50&amp;animation=fadeIn" alt="Advanced Information Retrieval Header"/>
</div>

---

# Advanced Information Retrieval with Distributional Semantics, Word Embeddings, and Learning-to-Rank

This project presents a comprehensive implementation of modern **Information Retrieval (IR)** techniques by integrating **distributional semantics**, **word embedding analysis**, and **Learning-to-Rank (LTR)** algorithms into a unified experimental framework. It investigates semantic word relationships through paradigmatic and syntagmatic analysis, explores pre-trained **GloVe** embeddings, and evaluates multiple ranking paradigms—including **Pointwise**, **Pairwise**, and **Listwise** approaches—on the Cranfield benchmark.

<div align="left">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat)](https://matplotlib.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Natural_Language_Processing-76B900?style=flat)](https://www.nltk.org/)
[![GloVe](https://img.shields.io/badge/GloVe-Word_Embeddings-E91E63?style=flat)](https://nlp.stanford.edu/projects/glove/)
[![Learning to Rank](https://img.shields.io/badge/Learning_to_Rank-Information_Retrieval-8E24AA?style=flat)](#)
[![Cranfield Dataset](https://img.shields.io/badge/Dataset-Cranfield-0F9D58?style=flat)](#)
[![Information Retrieval](https://img.shields.io/badge/Domain-Information_Retrieval-2563EB?style=flat)](#)
[![Semantic Analysis](https://img.shields.io/badge/NLP-Semantic_Analysis-14B8A6?style=flat)](#)
[![License](https://img.shields.io/badge/License-MIT-4B5563?style=flat)](https://opensource.org/licenses/MIT)

</div>

---

## Abstract

Information Retrieval systems rely heavily on effective document representation, semantic understanding, and ranking mechanisms to retrieve relevant information efficiently. This project provides an end-to-end exploration of modern Information Retrieval by combining **distributional semantics**, **semantic word embeddings**, and **Learning-to-Rank (LTR)** methodologies.

The project investigates lexical semantics through **Paradigmatic** and **Syntagmatic** word association analysis, analyzes semantic representations learned by **pre-trained GloVe embeddings** using PCA visualizations, and evaluates **Pointwise**, **Pairwise**, and **Listwise** ranking algorithms on the **Cranfield benchmark**. 

---

## Table of Contents

1. [Overview](#-overview)
2. [System Pipeline](#-system-pipeline)
3. [Distributional Semantics](#-distributional-semantics)
4. [Word Embeddings](#-word-embeddings)
5. [Learning-to-Rank Framework](#-learning-to-rank-framework)
6. [Experimental Results](#-experimental-results)
7. [Visualizations](#-visualizations)
8. [Project Structure](#-project-structure)
9. [Installation](#-installation)
10. [License & Author](#license)

---

# 📌 Overview

This project builds a comprehensive Information Retrieval experimentation framework by bridging traditional NLP techniques with machine learning-based ranking algorithms. Rather than focusing on a single retrieval model, it explores how semantic knowledge can be extracted from corpora (co-occurrence statistics and PMI), represented using dense embeddings (GloVe), and ultimately utilized in supervised ranking models (Pointwise, Pairwise, Listwise) for optimal document retrieval.

---

# ⚙️ System Pipeline

The project follows a multi-stage Information Retrieval pipeline that progressively transforms raw textual information into semantically meaningful representations before applying supervised ranking algorithms.

```mermaid
flowchart TB

subgraph Corpus
A[Text Collection]
end

subgraph Semantic Analysis
B[Tokenization]
C[Co-occurrence Matrix]
D[Paradigmatic Relations]
E[Syntagmatic Relations]
end

subgraph Embeddings
F[GloVe Embeddings]
G[Cosine Similarity]
H[PCA Visualization]
end

subgraph Learning to Rank
I[Feature Engineering]
J[Pointwise Models]
K[Pairwise Models]
L[Listwise Models]
end

subgraph Evaluation
M[NDCG]
N[MAP]
O[MRR]
P[Precision]
Q[Recall]
end

A --> B
B --> C
C --> D
C --> E
D --> F
E --> F
F --> G
G --> H
F --> I
I --> J
I --> K
I --> L
J --> M
K --> N
L --> O
J --> P
L --> Q
```

### Architectural Components

| Layer | Responsibility |
|------------|------------------------------|
| **Corpus** | Raw textual collection (Cranfield benchmark) |
| **Semantic Layer** | Distributional semantics, Paradigmatic & Syntagmatic relations |
| **Embedding Layer** | Dense semantic representation using GloVe and PCA |
| **Ranking Layer** | Supervised Learning-to-Rank algorithms |
| **Evaluation Layer** | Retrieval performance analysis using standard metrics |

---

# 📖 Distributional Semantics

Distributional Semantics is founded on the linguistic hypothesis that words appearing in similar contexts tend to share similar meanings. This project investigates two complementary forms of semantic relationships:

## 🔹 Paradigmatic Relations
Captures semantic similarity between words that can substitute for one another in similar contexts (e.g., *system* vs. *framework*).

```mermaid
flowchart LR
A[Corpus] --> B[Tokenization] --> C[Context Collection] --> D[Co-occurrence Matrix] --> E[Similarity Computation] --> F[Paradigmatic Neighbors]
```

## 🔹 Syntagmatic Relations
Describes words that frequently occur together within local contexts, exhibiting strong contextual dependency (e.g., *computer* + *software*).

```mermaid
flowchart LR
A[Corpus] --> B[Sliding Window] --> C[Word Co-occurrences] --> D[PMI Computation] --> E[Association Scores] --> F[Syntagmatic Relations]
```

---

# 🧠 Word Embeddings

Traditional sparse representations often fail to capture deeper semantic relationships among words. This project utilizes **pre-trained GloVe embeddings** to encode semantic information into dense continuous vector spaces, enabling:

- **Semantic Neighborhood Retrieval:** Identifying semantically related words (like technology-related words clustering around *computer*) using cosine similarity.
- **Embedding Space Visualization:** Projecting high-dimensional vectors into 2D spaces using **PCA** for visual inspection of clustering behavior and embedding geometry.

---

# 📈 Learning-to-Rank Framework

Instead of simply classifying documents as relevant or non-relevant, Learning-to-Rank models learn an ordering function that prioritizes highly relevant documents.

### Ranking Paradigms Implemented:
1. **Pointwise:** Formulates ranking as standard regression or classification. Evaluates each document independently (e.g., Random Forest, Gradient Boosting).
2. **Pairwise:** Transforms ranking into a binary preference problem, learning which document in a pair should be ranked higher (e.g., RankNet, RankSVM).
3. **Listwise:** Directly optimizes entire ranked lists and evaluation metrics such as NDCG (e.g., LambdaMART, ListNet).

### Ranking Pipeline

```mermaid
flowchart TB
A[Queries] --> B[Candidate Documents] --> C[Feature Extraction] --> D{Ranking Paradigm}
D --> E[Pointwise]
D --> F[Pairwise]
D --> G[Listwise]
E --> H[Ranked Documents]
F --> H
G --> H
H --> I[Evaluation Metrics]
```

*(Note: The performance of all ranking models is comprehensively evaluated on the Cranfield dataset using standard IR metrics including **NDCG@k**, **MAP**, **MRR**, **Precision@k**, and **Recall@k**.)*

---

# 📊 Experimental Results & Visualizations

Overall experimental results indicate that **Listwise** ranking algorithms provide the highest retrieval effectiveness, while **Pairwise** methods offer competitive performance with lower computational complexity. Furthermore, **GloVe** embeddings successfully capture meaningful semantic neighborhoods, verified by PCA projections.

### Visual Analysis
*(Visualizations are available in the `assets/images/` directory)*
- **Paradigmatic vs Syntagmatic Relations:** Compares similarity-based vs. contextual co-occurrence relationships.
- **PCA Visualizations:** Demonstrates semantic neighborhoods in low-dimensional space.
- **LTR Comparison & Heatmap:** Summarizes retrieval performance across all ranking models and metrics.

---

# 📁 Project Structure

```text
Information-Retrieval-with-Learning-to-Rank-and-Distributional-Semantics
│
├── IIR-CA4-810103099.ipynb
│
├── datasets/
│   ├── cran.all.1400
│   ├── cran.qry
│   ├── cranqrel
│   └── cranqrel.readme
│
├── assets/
│   └── images/
│       ├── embedding_pca.png
│       ├── paradigmatic_pca_visualization.png
│       ├── paradigmatic_vs_syntagmatic_comparison.png
│       ├── ltr_comparison.png
│       └── ltr_heatmap.png
│
├── outputs/
│   ├── embedding_similarities.csv
│   └── ltr_results.csv
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Clone Repository
```bash
git clone https://github.com/farzadjannati/Information-Retrieval-with-Learning-to-Rank-and-Distributional-Semantics.git
cd Information-Retrieval-with-Learning-to-Rank-and-Distributional-Semantics
```

## Create Environment & Run
```bash
conda create -n information_retrieval python=3.10
conda activate information_retrieval
pip install -r requirements.txt
jupyter notebook IIR-CA4-810103099.ipynb
```

---

# License

This project is licensed under the MIT License.

---

## Author

**Farzad Jannati**

M.Sc. Student, University of Tehran  
Research Assistant @ Social Networks Lab

**Research Interests:** Information Retrieval, NLP, Learning-to-Rank, LLMs, RAG, Agentic AI

📧 [farzadjannati@ut.ac.ir](mailto:farzadjannati@ut.ac.ir) | 💻 [github.com/farzadjannati](https://github.com/farzadjannati) | 💼 [linkedin.com/in/farzadjannati](https://www.linkedin.com/in/farzadjannati)

---

# ⭐ Support

If you find this repository useful, consider giving it a ⭐ on GitHub. Contributions and discussions are always welcome.

<p align="center">
Built with ❤️ using Python, Scikit-Learn, GloVe, NLTK, and Learning-to-Rank
</p>
