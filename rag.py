import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# 1. Загружаем ключ
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[!] Ошибка: добавь GEMINI_API_KEY в файл .env")
    exit()

from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.gemini import Gemini

# 2. Подключаем работающую модель Gemini
print("[*] Подключаем Gemini...")
llm = Gemini(model="models/gemini-flash-latest", api_key=api_key)

# 3. Читаем все 25 дисциплин из папки data/
print("[*] Читаем файлы дисциплин из data/...")
documents = SimpleDirectoryReader("data").load_data()
print(f"    Загружено файлов: {len(documents)}")

# 4. Нарезаем документы на логические абзацы (ноды)
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"    Получено текстовых фрагментов: {len(nodes)}")

# 5. Создаем локальный мгновенный поисковик BM25
# similarity_top_k=2 означает: берем 2 самых релевантных фрагмента
retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

# 6. Четкий промпт помощника первокурсника ВШЭ
qa_template_str = (
    "Ты — умный помощник первокурсника ИВТ МИЭМ НИУ ВШЭ по учебным дисциплинам.\n"
    "Используй ТОЛЬКО приведенный ниже контекст из программ курсов.\n"
    "Сформулируй краткий, точный и понятный ответ на вопрос студента.\n"
    "Если информации в контексте нет, честно ответь: 'В программе дисциплины нет информации по этому вопросу'.\n\n"
    "Контекст:\n{context_str}\n\n"
    "Вопрос: {query_str}\n\n"
    "Ответ (обязательно укажи в конце точную ссылку на источник из текста):"
)
qa_template = PromptTemplate(qa_template_str)

# 7. Собираем RAG движок: Поиск (BM25) -> Ответ (Gemini)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
    text_qa_template=qa_template
)

# 8. Тестовый вопрос
question = "Какая формула оценивания и элементы контроля по курсу Алгебра и геометрия?"
print(f"\n[?] Вопрос: {question}")
print("=" * 60)

response = query_engine.query(question)

print("\n[!] Ответ помощника:\n")
print(response.response)

print("\n" + "=" * 60)
print("[+] Использованные файлы:")
for node in response.source_nodes:
    print(f"- {node.metadata.get('file_name')}")