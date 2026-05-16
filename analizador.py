import os
import requests
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai


# Cargar variables de entorno desde el archivo .env
load_dotenv()

class AnalizadorFinanciero:
    def __init__(self):
        """
        Inicializa el Analizador Financiero obteniendo la API Key de las variables de entorno.
        """
        self.api_key = os.getenv("FMP_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://financialmodelingprep.com/stable"
        
        if not self.api_key:
            raise ValueError("FMP_API_KEY no está configurada.")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            # Usar Gemini 3 Flash Preview (disponible en 2026)
            self.model_name = 'gemini-3-flash-preview'
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None
            print("Advertencia: GOOGLE_API_KEY no configurada. El análisis IA estará desactivado.")

    def obtener_key_metrics(self, ticker: str, period: str = "annual", limit: int = 5) -> pd.DataFrame:
        """
        Obtiene las métricas clave (Key Metrics) de una empresa desde la API de FMP.
        
        Args:
            ticker (str): Símbolo de la acción (ej. 'AAPL').
            period (str): Periodo de los datos ('annual' o 'quarter').
            limit (int): Número de periodos a obtener.
            
        Returns:
            pd.DataFrame: DataFrame con los Key Metrics.
        """
        endpoint = f"{self.base_url}/key-metrics"
        params = {
            "symbol": ticker.upper(),
            "period": period,
            "limit": limit,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status() # Lanza una excepción si hay un error HTTP
            data = response.json()
            
            if not data:
                print(f"Advertencia: No se encontraron datos para el ticker {ticker.upper()}.")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            return df
            
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP al consultar la API para {ticker}: {http_err}")
            return pd.DataFrame()
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Error de conexión con la API: {conn_err}")
            return pd.DataFrame()
        except requests.exceptions.Timeout as timeout_err:
            print(f"Tiempo de espera agotado al consultar la API: {timeout_err}")
            return pd.DataFrame()
        except Exception as err:
            print(f"Error inesperado al obtener datos de la API: {err}")
            return pd.DataFrame()

    def obtener_ratios_salud(self, ticker: str) -> dict:
        """
        Calcula y retorna los ratios de salud financiera (estilo Buffett) usando Key Metrics.
        Incluye ROE, Deuda/EBITDA, y Márgenes si están disponibles.
        """
        df_metrics = self.obtener_key_metrics(ticker, limit=1)
        ratios = {
            "ROE": None,
            "Deuda_EBITDA": None,
            "Margen_Bruto": None,
            "Margen_Neto": None
        }
        
        if not df_metrics.empty:
            record = df_metrics.iloc[0]
            ratios["ROE"] = record.get("returnOnEquity")
            ratios["Deuda_EBITDA"] = record.get("netDebtToEBITDA")
        
        # Consultar income-statement para los márgenes
        endpoint_is = f"{self.base_url}/income-statement"
        params_is = {
            "symbol": ticker.upper(),
            "period": "annual",
            "limit": 1,
            "apikey": self.api_key
        }
        
        try:
            res = requests.get(endpoint_is, params=params_is)
            if res.status_code == 200:
                data_is = res.json()
                if data_is:
                    record_is = data_is[0]
                    revenue = record_is.get("revenue", 0)
                    gross_profit = record_is.get("grossProfit", 0)
                    net_income = record_is.get("netIncome", 0)
                    
                    if revenue and revenue > 0:
                        ratios["Margen_Bruto"] = gross_profit / revenue
                        ratios["Margen_Neto"] = net_income / revenue
        except Exception as e:
            print(f"Error al consultar income-statement para {ticker}: {e}")
            
        return ratios

    def calcular_valor_intrinseco_dcf(self, ticker: str) -> dict:
        """
        Obtiene el cálculo del Valor Intrínseco usando el modelo DCF proporcionado por FMP.
        """
        endpoint = f"{self.base_url}/discounted-cash-flow"
        params = {
            "symbol": ticker.upper(),
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"Advertencia: No se encontraron datos DCF para el ticker {ticker.upper()}.")
                return {}
                
            return data[0] # Retorna el primer registro (más reciente)
            
        except Exception as err:
            print(f"Error al obtener datos DCF de la API para {ticker}: {err}")
            return {}

    def obtener_precios_historicos(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Obtiene el historial de precios diarios usando la API de FMP (inmunidad total en Railway).
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker.upper()}?apikey={self.api_key}"
            response = requests.get(url)
            
            print("--- DEBUG FMP HISTORICO ---")
            print("Status Code:", response.status_code)
            print("Keys del JSON:", response.json().keys() if response.status_code == 200 else "No es 200")
            print("---------------------------")
            
            response.raise_for_status()
            data = response.json()
            
            if not data or "historical" not in data:
                print(f"Advertencia: No se encontraron datos históricos de FMP para {ticker.upper()}. Estructura recibida: {data}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data["historical"])
            if df.empty:
                return pd.DataFrame()
                
            # Invertir el orden para que vaya del más antiguo al más reciente
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Renombrar las columnas de minúsculas a Mayúsculas
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return df.tail(days)
            
        except Exception as e:
            print(f"Error al obtener precios con FMP: {e}")
            return pd.DataFrame()

    def identificar_soportes(self, df: pd.DataFrame, window: int = 20) -> list:
        """
        Identifica niveles de soporte basados en mínimos locales.
        """
        if df.empty or len(df) < window:
            return []
            
        # Encontrar mínimos locales
        lows = df["Low"].values
        soportes = []
        
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window : i+window]):
                soportes.append(round(float(lows[i]), 2))
        
        # Agrupar soportes cercanos (umbral del 2%) y contar frecuencia
        if not soportes:
            return []
            
        soportes.sort()
        clusters = []
        if soportes:
            current_cluster = [soportes[0]]
            for i in range(1, len(soportes)):
                if (soportes[i] - current_cluster[-1]) / current_cluster[-1] < 0.02:
                    current_cluster.append(soportes[i])
                else:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [soportes[i]]
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        # Retornar los 3 más relevantes (en este caso los últimos/más bajos detectados o simplemente los 3 primeros)
        return sorted(list(set([round(c, 2) for c in clusters])))[-3:]

    def calcular_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula el RSI (Relative Strength Index).
        """
        if df.empty or len(df) < period:
            return None
            
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Evitar división por cero
        loss = loss.replace(0, 0.001)
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(float(rsi.iloc[-1]), 2)

    def obtener_analisis_tecnico(self, ticker: str) -> dict:
        """
        Realiza un análisis técnico simplificado: RSI y Soportes.
        """
        df = self.obtener_precios_historicos(ticker)
        if df.empty:
            return {}

        rsi = self.calcular_rsi(df)
        soportes = self.identificar_soportes(df)
        
        endpoint = f"{self.base_url}/quote"
        params = {
            "symbol": ticker.upper(),
            "apikey": self.api_key
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            precio_actual = data[0]['price']
        except Exception as e:
            print(f"Error al obtener precio actual de FMP para {ticker}: {e}")
            precio_actual = float(df["Close"].iloc[-1])

        # Determinar mensaje RSI
        rsi_mensaje = "Neutral"
        if rsi is not None:
            if rsi < 30:
                rsi_mensaje = "Sobreventa / Oportunidad"
            elif rsi > 70:
                rsi_mensaje = "Sobrecompra / Riesgo"

        # Encontrar el soporte más cercano por debajo del precio actual
        soporte_cercano = None
        distancia_soporte = None
        
        soportes_debajo = [s for s in soportes if s < precio_actual]
        if soportes_debajo:
            soporte_cercano = max(soportes_debajo)
            distancia_soporte = round(((precio_actual - soporte_cercano) / precio_actual) * 100, 2)

        return {
            "rsi": rsi,
            "rsi_mensaje": rsi_mensaje,
            "soportes": soportes,
            "soporte_cercano": soporte_cercano,
            "distancia_soporte_porcentaje": distancia_soporte,
            "precio_actual": precio_actual
        }

    def obtener_noticias_recientes(self, ticker: str, limit: int = 5) -> list:
        """
        Obtiene las últimas noticias usando la API de FMP.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker.upper()}&limit={limit}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            news = response.json()
            
            if not news:
                return []
                
            textos = [f"Título: {n.get('title')}\nFuente: {n.get('site')}\nResumen: {n.get('text')}" for n in news[:limit]]
            return textos
            
        except Exception as e:
            print(f"Error al obtener noticias con FMP: {e}")
            return []

    def analizar_riesgos_ia(self, ticker: str) -> dict:
        """
        Utiliza Gemini para analizar riesgos, foso y sentimiento basado en noticias.
        """
        if not self.model:
            return {"error": "IA no configurada (falta GOOGLE_API_KEY)"}
            
        noticias_lista = self.obtener_noticias_recientes(ticker)
        texto_noticias = "\n\n---\n\n".join(noticias_lista) if noticias_lista else "No se encontraron noticias recientes. Basa el análisis en tu conocimiento general de los fundamentos y modelo de negocio de la empresa."
            
        prompt = f"""
        Actúa como un Analista de Riesgos Senior y experto en inversiones de Warren Buffett.
        Analiza los siguientes comunicados de prensa recientes de la empresa {ticker}:
        
        {texto_noticias}
        
        Basado en este texto y tu conocimiento general, proporciona un análisis estructurado:
        1. 3 Riesgos Estructurales (Amenazas críticas al modelo de negocio en los próximos 5 años).
        2. Moat (Foso): Identifica la ventaja competitiva real mencionada o implícita.
        3. Sentimiento (1-10): Calificación de la confianza de la directiva y tono general.
        
        Responde exclusivamente en formato JSON con la siguiente estructura:
        {{
            "riesgos_estructurales": ["Riesgo 1", "Riesgo 2", "Riesgo 3"],
            "moat": "Descripción breve del foso",
            "sentimiento_score": 8,
            "resumen_analista": "Una breve nota final"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Limpiar la respuesta para asegurar que sea JSON válido
            res_text = response.text
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0]
            
            import json
            return json.loads(res_text)
        except Exception as e:
            print(f"Error en análisis IA: {e}")
            return {"error": "Error procesando el análisis con la IA."}

# Ejemplo de prueba local
if __name__ == "__main__":
    try:
        print("Iniciando Analizador Financiero...")
        analizador = AnalizadorFinanciero()
        ticker = "AAPL"
        
        print(f"\n--- ANÁLISIS TÉCNICO DE {ticker} ---")
        tecnico = analizador.obtener_analisis_tecnico(ticker)
        print(tecnico)
        
        print(f"\n--- ANÁLISIS FUNDAMENTAL DE {ticker} ---")
        ratios = analizador.obtener_ratios_salud(ticker)
        print(ratios)
        
        dcf_data = analizador.calcular_valor_intrinseco_dcf(ticker)
        print(dcf_data)
        
        print(f"\n--- ANÁLISIS IA DE {ticker} ---")
        analisis_ia = analizador.analizar_riesgos_ia(ticker)
        print(analisis_ia)
        
    except Exception as e:
        print(f"Error en la ejecución: {e}")
