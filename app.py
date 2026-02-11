import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Корпоративный Поиск",
    page_icon="🔍",
    layout="wide"
)

# --- ЗАГРУЗКА МОДЕЛИ (КЭШИРОВАНИЕ) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

try:
    with st.spinner('Загрузка нейросети...'):
        model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# --- ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ---
def load_data():
    csv_file = "company_policies.csv"
    
    # Если есть CSV файл, грузим его
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # Превращаем DataFrame в список словарей для совместимости
            return df.to_dict('records')
        except Exception as e:
            st.error(f"Ошибка чтения CSV: {e}")
            return []
    
    # Иначе возвращаем дефолтные данные
    else:
        return [
            {"title": "Оформление отпуска", "content": "Для оформления ежегодного оплачиваемого отпуска необходимо подать заявление в HR-отдел за 2 недели.", "category": "HR"},
            {"title": "Настройка VPN", "content": "Для удаленного доступа используйте OpenVPN. Сервер: vpn.company.com.", "category": "IT"},
            {"title": "Дресс-код", "content": "Стиль Business Casual. По пятницам разрешен свободный стиль.", "category": "HR"},
            {"title": "Больничный", "content": "Сообщите номер электронного больничного руководителю.", "category": "HR"}
        ]

# Инициализация session_state для документов
if 'documents' not in st.session_state:
    st.session_state.documents = load_data()

# --- ФУНКЦИЯ ПОИСКА ---
def search(query, docs, top_k=3):
    if not docs:
        return []
        
    corpus = [doc['content'] for doc in docs]
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        doc_idx = int(idx)
        results.append({
            "score": float(score),
            "doc": docs[doc_idx]
        })
    return results

# --- ИНТЕРФЕЙС (FRONTEND) ---

with st.sidebar:
    st.header("⚙️ Управление")
    
    # Кнопка перезагрузки базы
    if st.button("🔄 Перезагрузить базу из CSV"):
        st.session_state.documents = load_data()
        st.success(f"Загружено {len(st.session_state.documents)} документов.")

    st.markdown("---")
    
    # Форма добавления
    with st.expander("➕ Добавить запись вручную"):
        new_title = st.text_input("Заголовок")
        new_cat = st.selectbox("Категория", ["HR", "IT", "Финансы", "Администрация", "Другое"])
        new_content = st.text_area("Текст")
        
        if st.button("Сохранить"):
            if new_title and new_content:
                st.session_state.documents.append({
                    "title": new_title,
                    "content": new_content,
                    "category": new_cat
                })
                st.success("Добавлено!")

    st.metric("Документов в индексе", len(st.session_state.documents))

st.title("🧠 Корпоративный Поиск")

# Проверка, откуда данные
if not os.path.exists("company_policies.csv"):
    st.info("💡 Совет: Загрузите файл `company_policies.csv` в репозиторий GitHub, чтобы использовать полную базу.")

query = st.text_input("Поисковый запрос:", placeholder="Например: потерял пропуск что делать?")

if query:
    with st.spinner('Поиск...'):
        results = search(query, st.session_state.documents)
    
    if not results:
        st.warning("Ничего не найдено.")
    
    for hit in results:
        score = hit['score']
        doc = hit['doc']
        
        # Цвет в зависимости от релевантности
        color = "#e6ffe6" if score > 0.6 else "#fffbe6" if score > 0.4 else "#fff0f0"
        
        with st.container():
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;">
                <div style="display:flex; justify-content:space-between;">
                    <h4 style="margin:0;">{doc.get('title', 'Без названия')}</h4>
                    <span style="background:#ddd; padding:2px 8px; border-radius:10px; font-size:0.8em;">{doc.get('category', 'Общее')}</span>
                </div>
                <p style="margin-top:10px;">{doc.get('content', '')}</p>
                <div style="font-size:0.8em; color:gray; margin-top:5px;">Релевантность: {score:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

with st.expander("📂 Посмотреть сырые данные"):
    st.dataframe(pd.DataFrame(st.session_state.documents))