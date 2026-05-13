import os
import requests
from dotenv import load_dotenv

load_dotenv()

def enviar_alerta_telegram(mensaje: str):
    """
    Envía un mensaje a un chat de Telegram a través de la Bot API.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("Error: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados.")
        return False
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error Telegram API: {response.text}")
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error al enviar mensaje por Telegram: {e}")
        return False
