# Utilisez une image de base contenant Python
FROM python:3.9

# Installez les dépendances Python
RUN pip install tensorflow numpy matplotlib scikit-learn jupyterlab

# Copiez le contenu de votre projet dans le conteneur
COPY . /app

# Définissez le répertoire de travail
WORKDIR /app

# Commande par défaut à exécuter lorsque le conteneur démarre
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
