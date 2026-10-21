import os
import re
import warnings
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("В файле .env не найден GEMINI_API_KEY!")
    st.stop()

import google.generativeai as genai
genai.configure(api_key=api_key, transport="rest")

from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.gemini import Gemini

st.set_page_config(
    page_title="Помощник первокурсника ИВТ МИЭМ",
    page_icon="🎓",
    layout="wide"
)

# Кэшируем инициализацию
@st.cache_resource(show_spinner="Инициализация базы данных...")
def init_rag():
    llm = Gemini(
        model="models/gemini-flash-lite-latest", 
        api_key=api_key, 
        transport="rest",
        temperature=0.0
    )
    documents = SimpleDirectoryReader("data").load_data()
    
    splitter = SentenceSplitter(chunk_size=6000, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)
    
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

    qa_template_str = (
        "Ты — умный помощник первокурсника ИВТ МИЭМ НИУ ВШЭ по учебным дисциплинам.\n"
        "Используй приведенный ниже контекст из официальных программ курсов.\n"
        "Правила ответа:\n"
        "1. Сформулируй краткий, точный и понятный ответ на вопрос студента.\n"
        "2. Отвечай строго по фактам из контекста. Делай естественные логические выводы "
        "(например: если написано, что все элементы контроля неблокирующие — прямо укажи, что блокирующих элементов нет).\n"
        "3. Если информации по теме вопроса в контексте действительно нет, честно ответь: "
        "'В программе дисциплины нет информации по этому вопросу'.\n\n"
        "Контекст:\n{context_str}\n\n"
        "Вопрос: {query_str}\n\n"
        "Ответ:"
    )
    qa_template = PromptTemplate(qa_template_str)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        text_qa_template=qa_template
    )

query_engine = init_rag()

with st.sidebar:
    st.title(" ИВТ МИЭМ 2026/2027")
    st.markdown("**База знаний:** 25 официальных программ дисциплин 1 курса.")
    st.markdown("---")
    st.subheader(" Примеры вопросов:")
    st.markdown("- Какие элементы контроля входят в Математический анализ?")
    st.markdown("- Есть ли блокирующие элементы по Алгебре и геометрии?")
    st.markdown("- Какая формула оценивания по курсу Алгебра и геометрия?")
    st.markdown("- Кто ведет курс Дискретная математика?")
    st.markdown("- Есть ли экзамен по Истории России?")
    st.markdown("---")
    if st.button("🗑️ Очистить диалог"):
        st.session_state.messages = []
        st.rerun()

st.title(" Помощник первокурсника по дисциплинам 1 курса")
st.caption("НИУ ВШЭ • ИВТ МИЭМ • 2026/2027 учебный год")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Я твой помощник по дисциплинам 1 курса ИВТ МИЭМ. Задай вопрос о формулах оценивания, блокирующих экзаменах, преподавателях или содержании предметов!"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Использованные первоисточники"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['title']}** — [Открыть на портале ВШЭ]({src['url']})")

if user_question := st.chat_input("Напишите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу информацию в программах курсов..."):
            resp = query_engine.query(user_question)

            unique_files = list(dict.fromkeys([node.metadata.get("file_name") for node in resp.source_nodes if node.metadata.get("file_name")]))
            
            sources = []
            for fname in unique_files:
                fpath = os.path.join("data", fname)
                course_title = fname.replace(".txt", "")
                course_title = re.sub(r"^\d+_", "", course_title)
                course_url = "https://www.hse.ru/ba/isct/courses"

                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = [f.readline().strip(), f.readline().strip()]
                        for line in lines:
                            if line.startswith("Дисциплина:"):
                                course_title = line.replace("Дисциплина:", "").strip()
                            elif line.startswith("Источник:"):
                                course_url = line.replace("Источник:", "").strip()

                sources.append({"title": course_title, "url": course_url})

            st.markdown(resp.response)

            if sources:
                with st.expander("Использованные первоисточники"):
                    for src in sources:
                        st.markdown(f"- **{src['title']}** — [Открыть на портале ВШЭ]({src['url']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.response,
                "sources": sources
            })