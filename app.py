from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# APIキー
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ベクトルDB初期化
def load_vectorstore():
    all_docs = []
    for filename in os.listdir("pdfs"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("pdfs", filename))
            pages = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)
            all_docs.extend(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(all_docs, embeddings)

db = load_vectorstore()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question", "")

    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    ブンブン！ハローYouTube!どうもヒカキンです！

    以下の情報を参考にして、質問に丁寧な日本語で答えてください。

    --- 参考情報 ---
    {context}

    --- 質問 ---
    {query}
    """

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return jsonify({"answer": response.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
