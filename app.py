import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Корпоративный Поиск",
    page_icon="🔍",
    layout="wide"
)

# --- ЗАГРУЗКА МОДЕЛИ ---
# Используем кэширование, чтобы не грузить модель каждый раз
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

try:
    with st.spinner('Загрузка нейросети (первый запуск может занять минуту)...'):
        model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# --- БАЗА ЗНАНИЙ (ВШИТА В КОД) ---
# Данные хранятся прямо тут, файлы не нужны
if 'documents' not in st.session_state:
    st.session_state.documents = [
        {"id": 1, "category": "HR", "title": "Оформление отпуска", "content": "Для оформления ежегодного оплачиваемого отпуска необходимо подать заявление в HR-отдел не позднее чем за 2 недели до начала. Заявление подписывается руководителем."},
        {"id": 2, "category": "IT", "title": "Настройка VPN", "content": "Для удаленного доступа к сети компании используйте клиент OpenVPN. Сервер: vpn.company.com. Логин и пароль как от компьютера."},
        {"id": 3, "category": "HR", "title": "Дресс-код", "content": "В компании принят стиль Business Casual. По пятницам разрешен свободный стиль одежды (джинсы, футболки)."},
        {"id": 4, "category": "Финансы", "title": "Квартальные отчеты", "content": "Финансовые отчеты сдаются до 5 числа месяца. Шаблоны лежат на диске Z в папке Finance."},
        {"id": 5, "category": "Офис", "title": "Заказ пропусков", "content": "Для заказа гостевого пропуска напишите на ресепшн за 3 часа до визита. Укажите ФИО и номер машины."},
        {"id": 6, "category": "IT", "title": "Почта на телефоне", "content": "Для настройки почты Outlook на iPhone используйте сервер mail.company.com и порт 993."},
        {"id": 7, "category": "Безопасность", "title": "Потеря пропуска", "content": "При утере пропуска срочно звоните в охрану по номеру 1122 для блокировки доступа."},
        {"id": 8, "category": "Бухгалтерия", "title": "Выплата зарплаты", "content": "Аванс выплачивается 20-го числа, основная часть зарплаты - 5-го числа следующего месяца."}
    ]

# --- ПОИСК ---
def search(query, docs, top_k=3):
    # Берем только тексты
    corpus = [doc['content'] for doc in docs]
    
    # Превращаем запрос и документы в векторы
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # Считаем схожесть
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Сортируем
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            "score": float(score),
            "doc": docs[int(idx)]
        })
    return results

# --- ИНТЕРФЕЙС ---
st.title("🧠 AI Поиск по базе знаний")

# Боковая панель для добавления
with st.sidebar:
    st.header("Добавить документ")
    new_title = st.text_input("Название")
    new_cat = st.selectbox("Категория", ["HR", "IT", "Финансы", "Офис"])
    new_content = st.text_area("Текст правила")
    if st.button("Добавить"):
        st.session_state.documents.append({
            "id": len(st.session_state.documents)+1,
            "title": new_title,
            "category": new_cat,
            "content": new_content
        })
        st.success("Добавлено!")

# Поиск
query = st.text_input("Что искать?", placeholder="Например: когда придет зарплата?")

if query:
    results = search(query, st.session_state.documents)
    
    if not results:
        st.write("Ничего не найдено.")
    
    for hit in results:
        doc = hit['doc']
        score = hit['score']
        
        # Красивый вывод
        st.markdown(f"""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between;">
                <b>{doc['title']}</b>
                <span style="background:#eee; padding: 2px 8px; border-radius: 5px; font-size: small;">{doc['category']}</span>
            </div>
            <p style="margin: 5px 0;">{doc['content']}</p>
            <small style="color: grey;">Совпадение: {int(score*100)}%</small>
        </div>
        """, unsafe_allow_html=True)