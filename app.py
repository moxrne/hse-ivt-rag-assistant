import os
import re
import warnings
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Загружаем окружение
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Настройка страницы в браузере
st.set_page_config(
    page_title="Помощник первокурсника ИВТ МИЭМ",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Помощник первокурсника ИВТ МИЭМ")
st.caption("НИУ ВШЭ • Программы дисциплин 1 курса (2026/2027) • RAG + Gemini")

# Инициализируем RAG-систему один раз (кэшируем, чтобы не пересоздавать при каждом клике)
@st.cache_resource(show_spinner="Загрузка базы знаний дисциплин...")
def init_rag():
    from llama_index.core import SimpleDirectoryReader, PromptTemplate
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.llms.gemini import Gemini

    llm = Gemini(model="models/gemini-flash-latest", api_key=api_key)
    
    # Считываем наши 25 файлов
    documents = SimpleDirectoryReader("data").load_data()
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

    qa_template_str = (
        "Ты — умный помощник первокурсника ИВТ МИЭМ НИУ ВШЭ по учебным дисциплинам.\n"
        "Используй ТОЛЬКО приведенный ниже контекст из официальных программ дисциплин.\n"
        "Сформулируй краткий, точный и понятный ответ на вопрос студента.\n"
        "Если информации в контексте нет, честно ответь: 'В программе дисциплины нет информации по этому вопросу'.\n\n"
        "Контекст:\n{context_str}\n\n"
        "Вопрос: {query_str}\n\n"
        "Ответ:"
    )
    qa_template = PromptTemplate(qa_template_str)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        text_qa_template=qa_template
    )
    return query_engine

query_engine = init_rag()

# История сообщений в чате
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Я твой помощник по дисциплинам 1 курса ИВТ МИЭМ. Спроси меня о формулах оценивания, содержании курсов или литературе!"}
    ]

# Отображаем переписку
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Первоисточники"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['title']}** [Ссылка на сайт ВШЭ]({src['url']})")

# Поле ввода вопроса
if user_question := st.chat_input("Например: Как оценивается Математический анализ?"):
    # Сохраняем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("Ищу информацию в программах дисциплин..."):
            response = query_engine.query(user_question)
            
            # Извлекаем ссылки на первоисточники из найденных кусков текста
            sources = []
            for node in response.source_nodes:
                text_chunk = node.text
                # Ищем заголовок и ссылку, которые мы сохраняли в файле
                match_title = re.search(r"Дисциплина:\s*(.*)", text_chunk)
                match_url = re.search(r"Источник:\s*(https?://[^\s]+)", text_chunk)
                
                title = match_title.group(1).strip() if match_title else node.metadata.get("file_name", "Курс")
                url = match_url.group(1).strip() if match_url else "https://www.hse.ru/ba/isct/courses"
                
                if {"title": title, "url": url} not in sources:
                    sources.append({"title": title, "url": url})

            # Выводим текст ответа
            st.markdown(response.response)
            
            # Выводим плашку с источниками
            if sources:
                with st.expander("📚 Первоисточники"):
                    for src in sources:
                        st.markdown(f"- **{src['title']}** [Открыть на портале ВШЭ]({src['url']})")

            # Запоминаем в истории
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response.response,
                "sources": sources
            })