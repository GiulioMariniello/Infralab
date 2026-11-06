# 1. Partiamo da un'immagine Python ufficiale e leggera
# (allineata alla versione usata in sviluppo: i pin di pandas/numpy richiedono >=3.11)
FROM python:3.12-slim

# 2. Impostiamo la nostra directory di lavoro dentro il container
WORKDIR /app

# 3. Copiamo prima il file dei requisiti per ottimizzare i tempi di build futuri
COPY requirements.txt .

# 4. Installiamo tutte le librerie Python necessarie
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamo tutto il resto della nostra applicazione
COPY . .

# --- CORREZIONE PER PERMESSI ---
# Imposta le variabili d'ambiente per forzare Streamlit e Matplotlib
# a usare una cartella locale all'app, dove abbiamo i permessi.
ENV STREAMLIT_HOME=/app/config/streamlit
ENV MPLCONFIGDIR=/app/config/matplotlib

# Crea le directory necessarie e assegna i permessi
RUN mkdir -p /app/config/streamlit /app/config/matplotlib /app/data/sessions /app/data/free_play && \
    chmod -R 777 /app/config /app/data

# 6. Diciamo a Docker che la nostra app funziona sulla porta 7860 (standard HF Spaces)
EXPOSE 7860

# 7. Il comando per avviare l'app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
