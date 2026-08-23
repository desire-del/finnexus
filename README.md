# Finexus

## Interface Streamlit

Lancez l’interface de recherche SEC depuis la racine du projet :

```bash
uv run streamlit run streamlit_app.py
```

L’application initialise le runtime RAG une seule fois, conserve les ressources
asynchrones entre les reruns Streamlit et utilise les providers configurés dans
le fichier `.env`. La vue `Filings` permet d’ingérer le dernier dépôt SEC d’un
ticker ou CIK et d’afficher les filings disponibles pour le profil d’embedding
actif.
