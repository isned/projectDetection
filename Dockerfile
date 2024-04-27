# Utiliser l'image Python officielle de base
FROM python:3.8-slim

# Mettre à jour la liste des paquets et installer les dépendances nécessaires
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requis dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application dans le conteneur
COPY . .

# Exposer le port 8501
EXPOSE 8501

# Commande par défaut pour lancer l'application Streamlit
CMD ["streamlit", "run", "detect_gender_webcam.py"]
