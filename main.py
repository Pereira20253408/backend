from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from database import get_db
from analizador import AnalizadorFinanciero
from notifications import enviar_alerta_telegram
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI(
    title="Finanza API",
    description="API para análisis de inversiones a largo plazo",
    version="1.1.0"
)

# Modelos Pydantic
class WatchlistItem(BaseModel):
    ticker: str
    precio_objetivo: float
    soporte_tecnico: float
    fecha_analisis: str = str(datetime.now().date())
    seguimiento_activo: bool = True
    precio_actual: float = 0.0
    rsi: float = 0.0

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciar el analizador
try:
    analizador = AnalizadorFinanciero()
except Exception as e:
    print(f"Error al inicializar el Analizador Financiero: {e}")
    analizador = None

db = get_db()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bienvenido a la API de Finanza"}

@app.get("/analizar/{ticker}")
def obtener_analisis_completo(ticker: str):
    """
    Análisis financiero completo: Health, DCF y Technical.
    """
    if not analizador:
        raise HTTPException(status_code=500, detail="El analizador no está configurado.")
        
    try:
        # 1. Fundamental
        ratios = analizador.obtener_ratios_salud(ticker)
        dcf_data = analizador.calcular_valor_intrinseco_dcf(ticker)
        
        # 2. Técnico
        tecnico = analizador.obtener_analisis_tecnico(ticker)

        if not ratios and not dcf_data and not tecnico:
            raise HTTPException(status_code=404, detail="Ticker no encontrado.")

        respuesta = {
            "ticker": ticker.upper(),
            "ratios_salud": {
                "roe": ratios.get("ROE"),
                "deuda_ebitda": ratios.get("Deuda_EBITDA"),
                "margen_bruto": ratios.get("Margen_Bruto"),
                "margen_neto": ratios.get("Margen_Neto")
            },
            "valor_intrinseco": {
                "dcf": dcf_data.get("dcf"),
                "precio_actual": dcf_data.get("Stock Price") or tecnico.get("precio_actual"),
                "fecha": dcf_data.get("date")
            },
            "analisis_tecnico": tecnico,
            "fecha_consulta": str(datetime.now())
        }
        
        # Lógica de veredicto
        veredicto = "Esperar"
        if respuesta["valor_intrinseco"]["dcf"] and respuesta["valor_intrinseco"]["precio_actual"]:
            dcf = respuesta["valor_intrinseco"]["dcf"]
            precio = respuesta["valor_intrinseco"]["precio_actual"]
            margen = (dcf - precio) / dcf
            respuesta["valor_intrinseco"]["margen_seguridad_porcentaje"] = round(margen * 100, 2)
            
            if precio < dcf:
                if margen >= 0.20: veredicto = "Comprar (Gran Oportunidad)"
                elif margen >= 0.10: veredicto = "Comprar"
                else: veredicto = "Mantener / Vigilar"
        
        respuesta["veredicto_final"] = veredicto
        return respuesta

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analizar-ia/{ticker}")
def obtener_analisis_ia(ticker: str):
    """
    Análisis de riesgos y cualitativo mediante IA (Gemini).
    """
    if not analizador:
        raise HTTPException(status_code=500, detail="El analizador no está configurado.")
    
    try:
        resultado = analizador.analizar_riesgos_ia(ticker)
        if "error" in resultado:
            raise HTTPException(status_code=400, detail=resultado["error"])
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS WATCHLIST ---

@app.post("/watchlist")
def add_to_watchlist(item: WatchlistItem):
    if not db:
        raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        item_dict = item.dict()
        if item.precio_actual == 0.0 or item.rsi == 0.0:
            if analizador:
                tecnico = analizador.obtener_analisis_tecnico(item.ticker)
                if tecnico:
                    item_dict["precio_actual"] = tecnico.get("precio_actual", item.precio_actual)
                    item_dict["rsi"] = tecnico.get("rsi", item.rsi)
                    
        db.collection("watchlist").document(item.ticker.upper()).set(item_dict)
        return {"message": f"{item.ticker} añadido a la watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/watchlist")
def get_watchlist():
    if not db:
        raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        docs = db.collection("watchlist").stream()
        watchlist = [doc.to_dict() for doc in docs]
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/watchlist/{ticker}")
def eliminar_de_watchlist(ticker: str):
    if not db:
        raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        db.collection("watchlist").document(ticker.upper()).delete()
        return {"message": f"{ticker} eliminado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/watchlist/{ticker}/toggle")
def toggle_seguimiento(ticker: str):
    if not db:
        raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        doc_ref = db.collection("watchlist").document(ticker.upper())
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Ticker no encontrado en la watchlist.")
        
        data = doc.to_dict()
        estado_actual = data.get("seguimiento_activo", True)
        nuevo_estado = not estado_actual
        
        doc_ref.update({"seguimiento_activo": nuevo_estado})
        return {"ticker": ticker.upper(), "seguimiento_activo": nuevo_estado, "message": f"Seguimiento para {ticker.upper()} cambiado a {nuevo_estado}"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- LÓGICA DEL VIGILANTE ---

def tarea_vigilancia():
    """
    Revisa la watchlist y envía alertas si el precio está cerca de un soporte.
    """
    print(f"[{datetime.now()}] Iniciando tarea de vigilancia...")
    if not db or not analizador:
        return

    try:
        docs = db.collection("watchlist").where("seguimiento_activo", "==", True).stream()
        for doc in docs:
            item = doc.to_dict()
            ticker = item["ticker"]
            soporte = item["soporte_tecnico"]
            precio_objetivo = item["precio_objetivo"]

            # Obtener datos técnicos actuales
            tecnico = analizador.obtener_analisis_tecnico(ticker)
            if not tecnico: continue

            precio_actual = tecnico["precio_actual"]
            rsi = tecnico["rsi"]
            
            # Guardar/Actualizar en Firestore el precio_actual y rsi
            db.collection("watchlist").document(ticker.upper()).update({
                "precio_actual": precio_actual,
                "rsi": rsi,
                "ultima_actualizacion": str(datetime.now())
            })
            
            # Calcular distancia al soporte
            if soporte > 0:
                distancia = (precio_actual - soporte) / soporte
                
                # Si está a un 2% o menos del soporte
                if 0 <= distancia <= 0.02:
                    mensaje = (
                        f"🚨 *ALERTA DE COMPRA: {ticker}*\n\n"
                        f"El precio está en zona de soporte técnico.\n"
                        f"💰 *Precio Actual:* ${precio_actual:.2f}\n"
                        f"📉 *Soporte:* ${soporte:.2f} (Distancia: {distancia*100:.2f}%)\n"
                        f"🎯 *Fair Price (DCF):* ${precio_objetivo:.2f}\n"
                        f"📊 *RSI:* {rsi}\n\n"
                        f"La acción está en un punto técnico ideal de entrada."
                    )
                    enviar_alerta_telegram(mensaje)
                    print(f"Alerta enviada para {ticker}")

    except Exception as e:
        print(f"Error en la tarea de vigilancia: {e}")

# Configuración del Scheduler
scheduler = BackgroundScheduler()
# Ejecutar cada 4 horas de lunes a viernes
scheduler.add_job(tarea_vigilancia, 'cron', day_of_week='mon-fri', hour='0,4,8,12,16,20')

@app.on_event("startup")
def start_scheduler():
    scheduler.start()
    print("Scheduler iniciado (Vigilancia activa cada 4 horas).")
