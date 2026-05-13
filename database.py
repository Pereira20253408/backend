import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# Inicialización de Firebase
# Soporta tanto un archivo físico como una variable de entorno con el JSON completo
cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

try:
    if not firebase_admin._apps:
        if cred_json:
            # Cargar desde variable de entorno (ideal para Railway/Production)
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        elif os.path.exists(cred_path):
            # Cargar desde archivo local
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            # Fallback (puede requerir GOOGLE_APPLICATION_CREDENTIALS)
            firebase_admin.initialize_app()
    
    db = firestore.client()
    print("Firebase inicializado correctamente.")
except Exception as e:
    print(f"Error al inicializar Firebase: {e}")
    db = None

def get_db():
    return db
