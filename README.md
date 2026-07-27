# ECHO: Explainable Sequential Recommendation System 🛍️

> *Building smarter recommendations by combining sequential user behavior with semantic understanding.*

ECHO is a deep learning–based **sequential recommendation system** that predicts the next item a user is likely to purchase by combining **behavioral patterns** with **semantic product understanding**.

Unlike traditional recommendation systems that rely only on interaction history, ECHO enriches sequential user modeling with **pretrained BGE-Large semantic embeddings** and aligns both representations through **contrastive learning (InfoNCE)**. This enables the model to learn richer user representations while also providing interpretable recommendations.

---

# 📌 Problem Statement

Sequential recommendation models such as **GRU4Rec** learn user behavior from historical purchase sequences.

However, interaction history alone often lacks semantic information.

For example,

```
Laptop
↓

Mouse
↓

Keyboard
```

contains a strong semantic relationship that cannot be inferred from item IDs alone.

ECHO addresses this by combining

- **Sequential behavioral learning**
- **Semantic representation learning**

to generate more meaningful recommendations.

---

# 🧠 Architecture

```
                    Amazon Beauty Dataset
                             │
               ┌─────────────┴─────────────┐
               │                           │
         Item Purchase IDs          Product Titles
               │                           │
               ▼                           ▼
        Item Embedding Layer        BGE-Large Encoder
               │                           │
               ▼                           ▼
            GRU4Rec              1024-D Semantic Embeddings
               │                           │
               └─────────────┬─────────────┘
                             ▼
                InfoNCE Contrastive Learning
                             │
                             ▼
              Shared 128-D User Representation
                             │
                             ▼
                 Next-Item Recommendation
```

---

# ✨ Features

- Sequential recommendation using **GRU4Rec**
- Semantic embedding generation using **BAAI/bge-large-en-v1.5**
- Contrastive representation learning with **InfoNCE**
- Shared latent space learning between semantic and sequential representations
- Explainable recommendation pipeline
- Top-K recommendation inference
- Evaluation using **HR@K** and **NDCG@K**

---

# 🔬 Machine Learning Pipeline

### 1. Data Preprocessing

- Load Amazon Beauty purchase histories
- Convert Item IDs into product titles
- Create long-term and short-term user histories
- Generate train/test sequences using leave-one-out evaluation

---

### 2. Semantic Embedding Generation

Each user's purchase history is converted into natural language and encoded using

```
BAAI/bge-large-en-v1.5
```

This produces

- Long-term semantic embeddings
- Short-term semantic embeddings

Each embedding has

```
1024 dimensions
```

---

### 3. Sequential User Modeling

User purchase sequences are processed using

```
GRU4Rec
```

The model learns

- sequential purchasing behavior
- evolving user interests
- next-item prediction

---

### 4. Representation Alignment

Semantic embeddings are projected into a shared

```
128-dimensional latent space
```

and aligned with GRU hidden representations using

```
InfoNCE Contrastive Loss
```

This encourages semantic and sequential representations to capture similar user preferences.

---

### 5. Recommendation Generation

The trained model predicts

```
Top-K
```

products that the user has not previously interacted with.

---

# 📊 Results

Evaluation was performed on the **Amazon Beauty Dataset**.

| Metric | GRU4Rec Baseline | ECHO |
|---------|-----------------:|------:|
| HR@5 | 0.0153 | **0.0229** |
| HR@10 | 0.0246 | **0.0330** |
| HR@20 | 0.0390 | **0.0427** |
| NDCG@5 | 0.0087 | **0.0149** |
| NDCG@10 | 0.0117 | **0.0181** |
| NDCG@20 | 0.0143 | **0.0206** |

Compared to the GRU4Rec baseline:

- **34% improvement in HR@10**
- **55% improvement in NDCG@10**

---

# 🛠 Tech Stack

## Machine Learning

- Python
- PyTorch
- GRU4Rec
- Sentence Transformers
- BAAI/bge-large-en-v1.5

## Evaluation

- HR@K
- NDCG@K

## Backend

- FastAPI

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

---

# 📁 Project Structure

```
ECHO
│
├── ml/
│   ├── generate_embeddings.py
│   ├── train_gru4rec.py
│   ├── evaluate.py
│   ├── inference.py
│   └── gru4rec_best.pth
│
├── backend/
│
├── frontend/
│
├── Beauty/
│
└── README.md
```

---

# 🚀 Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Generate Semantic Embeddings

```bash
cd ml

python generate_embeddings.py
```

---

## Train the Recommendation Model

```bash
python train_gru4rec.py
```

---

## Evaluate

```bash
python evaluate.py
```

---

## Start Backend

```bash
uvicorn backend.main:app --reload
```

---

## Start Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 📚 Key Concepts Demonstrated

- Sequential Recommendation
- Representation Learning
- Contrastive Learning
- Semantic Embeddings
- Recommendation Systems
- Explainable AI
- Deep Learning
- GRU Networks
- Sentence Transformers
- Information Retrieval

---

# Future Improvements

- Transformer-based recommendation models (SASRec / BERT4Rec)
- Multi-modal recommendation using product images
- Real-time online user embedding updates
- Approximate nearest-neighbor retrieval using FAISS
- Larger-scale recommendation datasets

---

# Author

**Hiya Modi**

Built as an end-to-end deep learning project exploring the intersection of sequential recommendation, semantic representation learning, and explainable AI.
