from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch

components_initialized = False
emotion_client = None
vectorstore = None
llm = None
prompt_template = None

app = FastAPI(title="ChatRealms API", description="Emotionally adaptive AI chat system", version="1.0.0")

class ChatRequest(BaseModel):
    texts: List[str]

def initialize_components():
    global components_initialized, emotion_client, vectorstore, llm, prompt_template

    if components_initialized:
        return

    print("🔄 Initializing components...")

    from gradio_client import Client
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    emotion_client = Client("culer555/bert-emotion-api", verbose=False)

    loader = TextLoader("chatrealms_context.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipe)

    prompt_template = PromptTemplate(
        input_variables=["context", "emotion", "user_input"],
        template=(
            "You are ChatRealms — a friendly, emotionally aware AI system.\n"
            "Context: {context}\n\n"
            "User emotion: {emotion}\n"
            "User message: {user_input}\n\n"
            "Respond with empathy and a natural tone that matches the emotion.\n"
            "Keep it under 3 sentences.\n"
            "Answer:"
        )
    )

    components_initialized = True
    print("✅ All components initialized!")

def get_emotion(text: str):
    try:
        result = emotion_client.predict(x=text, api_name="/lambda")
        if isinstance(result, dict) and "emotion" in result:
            return result["emotion"]
        return str(result)
    except Exception:
        return "neutral"

@app.post("/chat")
def chat(request: ChatRequest):
    if not components_initialized:
        initialize_components()

    results = []
    for text in request.texts:
        emotion = get_emotion(text)
        docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(text)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = prompt_template.format(context=context, emotion=emotion, user_input=text)
        ai_response = llm.invoke(prompt)

        if isinstance(ai_response, str) and "Answer:" in ai_response:
            ai_response = ai_response.split("Answer:", 1)[-1].strip()

        results.append({"text": text, "emotion": emotion, "response": ai_response})

    return {"results": results}

@app.get("/")
def root():
    return {"status": "running", "message": "ChatRealms API is live!"}
