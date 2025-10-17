import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import plotly.express as px
import time

# --- Configuración inicial ---
st.set_page_config(page_title="Demo TF-IDF en Español", page_icon="🔍", layout="wide")

st.title("🔍 Demo TF-IDF en Español (Versión Mejorada)")

# Descargar stopwords de NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words("spanish"))

# --- Documentos por defecto ---
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in stop_words]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    suggested = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]
    for q in suggested:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

# --- Botón principal ---
if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        with st.spinner("Analizando similitud... ⏳"):
            time.sleep(0.6)
            
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                min_df=1
            )
            
            X = vectorizer.fit_transform(documents)
            
            # --- Mostrar matriz TF-IDF ---
            st.markdown("### 📊 Matriz TF-IDF")
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(documents))]
            )
            st.dataframe(df_tfidf.round(3), use_container_width=True)

            # --- Palabras más relevantes por documento ---
            st.markdown("### 🔍 Palabras más relevantes por documento")
            for i, doc_vector in enumerate(X.toarray()):
                top_indices = doc_vector.argsort()[-3:][::-1]
                top_words = [vectorizer.get_feature_names_out()[j] for j in top_indices]
                st.write(f"**Doc {i+1}:** {', '.join(top_words)}")

            # --- Calcular similitud ---
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()
            
            # --- Gráfico de similitud ---
            df_sim = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Similitud": similarities
            })
            fig = px.bar(
                df_sim,
                x="Documento",
                y="Similitud",
                text="Similitud",
                color="Similitud",
                color_continuous_scale="tealgrn",
                title="Similitud entre la pregunta y los documentos"
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # --- Mostrar mejor coincidencia ---
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]
            
            # Resaltado simple de palabras de la pregunta
            pattern = r'(' + '|'.join(re.escape(w) for w in question.lower().split()) + r')'
            highlighted = re.sub(pattern, r'**\1**', best_doc.lower())
            
            st.markdown("### 🎯 Respuesta")
            st.markdown(f"**Tu pregunta:** {question}")
            
            if best_score > 0.01:
                st.success(f"**Respuesta más similar:** {highlighted}")
                st.info(f"📈 Similitud: {best_score:.3f}")
            else:
                st.warning(f"**Respuesta (baja confianza):** {highlighted}")
                st.info(f"📉 Similitud: {best_score:.3f}")

            # --- Descarga de resultados ---
            csv = df_tfidf.round(3).to_csv().encode("utf-8")
            st.download_button(
                label="💾 Descargar matriz TF-IDF (CSV)",
                data=csv,
                file_name="matriz_tfidf.csv",
                mime="text/csv"
            )

# --- Estilos visuales ---
st.markdown("""
<style>
    .stApp { background-color: #e9edc9; color: #3a5a40; }
    .stTextInput > div > div > input { background-color: #fefae0; }
    .stTextArea textarea { background-color: #fefae0; }
</style>
""", unsafe_allow_html=True)
