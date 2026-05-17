import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# Inicialización de Firebase orientada a entorno 100% local con archivo JSON físico
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

try:
    if not firebase_admin._apps:
        if os.path.exists(cred_path):
            # Priorizar carga desde archivo local (entorno local)
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print(f"Firebase inicializado correctamente usando archivo local: {cred_path}")
        elif cred_json:
            # Fallback a variable de entorno
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase inicializado usando variable de entorno FIREBASE_CREDENTIALS_JSON.")
        else:
            # Fallback por defecto de Google Cloud
            firebase_admin.initialize_app()
            print("Firebase inicializado usando credenciales por defecto de Google Cloud.")
    
    db = firestore.client()
except Exception as e:
    print(f"Error crítico al inicializar Firebase Admin SDK: {e}")
    db = None

def get_db():
    return db
