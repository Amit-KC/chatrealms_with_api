from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from gradio_client import Client

emotion_client = Client("culer555/bert-emotion-api")

def get_emotion(text: str) -> str:
    try:
        result = emotion_client.predict(x=text, api_name="/lambda")
        if isinstance(result, dict) and "emotion" in result:
            return result["emotion"]
        return str(result)
    except Exception as e:
        print("⚠️ Emotion API error:", e)
        return "neutral"


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

loader = TextLoader("chatrealms_context.txt", encoding="utf-8")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     
    chunk_overlap=150,    
    separators=["\n\n---\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)

docs = splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cpu")

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.9,
    top_p=0.9,
)
llm = HuggingFacePipeline(pipeline=llm_pipe)

prompt_template = PromptTemplate(
    input_variables=["context", "emotion", "user_input"],
    template=(
        "You are ChatRealms — a friendly, emotionally aware AI system.\n"
        "Here’s some helpful background context about how ChatRealms works:\n"
        "{context}\n\n"
        "User emotion: {emotion}\n"
        "User message: {user_input}\n\n"
        "Respond with empathy and natural tone that matches the user's emotion.\n"
        "Keep your reply under 3 sentences, avoid restating the prompt"
        "Answer:"
    )
)


app = FastAPI(title="ChatRealms API")

class ChatRequest(BaseModel):
    texts: List[str]

@app.post("/chat")
def chat(request: ChatRequest):
    responses = []
    for text in request.texts:
        detected_emotion = get_emotion(text)

        docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(text)
        context = "\n\n".join([doc.page_content for doc in docs])

        formatted_prompt = prompt_template.format(
            context=context,
            emotion=detected_emotion,
            user_input=text
        )

        ai_reply = llm.invoke(formatted_prompt)

        if "Answer:" in ai_reply:
            ai_reply = ai_reply.split("Answer:", 1)[-1].strip()

        responses.append({
            "text": text,
            "detected_emotion": detected_emotion,
            "response": ai_reply
        })

    return {"results": responses}
