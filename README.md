Perfect, Tharun! Since you didn’t add a `README.md` during repo creation, no problem — we’ll add it manually now.

---

## ✅ Step-by-Step: Add `README.md` to Your Local Repo & Push

### 📄 Step 1: Create the `README.md`

In your project folder, create a new file named:

```
README.md
```

Paste this full content inside:

---

```markdown
# 🛍️ AI-Powered Product Search and Management App

This Streamlit-based application enables semantic product search, intelligent product embedding using OpenAI, and product data management using Pinecone for vector search and JSON for persistence. It’s designed to handle large-scale product listings and deliver relevant search results using natural language queries.

## 🚀 Features

- **Semantic Product Search**: Uses OpenAI embeddings and Pinecone to find relevant products from natural language queries.
- **Product Management Dashboard**: View, edit, and resync product details and embeddings.
- **AI Query Understanding**: Analyzes the intent behind search queries to improve product retrieval.
- **Spell Correction & Synonym Matching**: Improves user search experience using NLP.
- **Batch Processing**: Efficiently handles large product datasets.
- **Extensible Architecture**: Modular design for easy extension (e.g., adding image embeddings or filters).

---

## 🧰 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Embedding Model**: `text-embedding-ada-002` (OpenAI)
- **Vector DB**: [Pinecone](https://www.pinecone.io/)
- **NLP Utilities**: NLTK, Hugging Face Transformers, Sentence Transformers, SpellChecker

---

## 📂 Project Structure

```plaintext
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .streamlit/secrets.toml     # API secrets (ignored from Git)
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Tharunbaikani/ai-product-search-app.git
cd ai-product-search-app
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔐 Setup API Keys

Create the file `.streamlit/secrets.toml` and paste this:

```toml
openai_api_key = "your_openai_api_key"
pinecone_api_key = "your_pinecone_api_key"
```

> ⚠️ `secrets.toml` is listed in `.gitignore` and should NOT be committed to GitHub.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Loads and embeds product data** using OpenAI’s embedding model.
2. **Stores vector data** in Pinecone.
3. **Uses GPT** to understand query context.
4. **Performs semantic search** with advanced scoring logic (name, category, tags, etc.).
5. **Displays results** in a clean Streamlit interface.

---

## 🧪 Try a Query

Use the chatbot to test queries like:

> _"biodegradable tissue for office washrooms"_

> _"hand sanitizers for schools with aloe vera"_

---


## 👨‍💻 Author

**Tharun**  
*AI/ML Enthusiast | Building intelligent tools*
```



