import os
import requests
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta


# Cargar variables de entorno desde el archivo .env
load_dotenv()

class AnalizadorFinanciero:
    def __init__(self):
        """
        Inicializa el Analizador Financiero obteniendo las API Keys de las variables de entorno.
        """
        self.finnhub_token = os.getenv("FINNHUB_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://finnhub.io/api/v1"
        
        if not self.finnhub_token:
            print("Advertencia: FINNHUB_API_KEY no está configurada.")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.model_name = 'gemini-3-flash-preview'
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None
            print("Advertencia: GOOGLE_API_KEY no configurada. El análisis IA estará desactivado.")

    def obtener_ratios_salud(self, ticker: str) -> dict:
        """
        Calcula y retorna los ratios de salud financiera usando Finnhub.io.
        Incluye ROE, Deuda/Capital, y Márgenes.
        """
        ratios = {
            "ROE": None,
            "Deuda_EBITDA": None,
            "Margen_Bruto": None,
            "Margen_Neto": None
        }
        
        if not self.finnhub_token:
            return ratios
            
        endpoint = f"{self.base_url}/stock/metric"
        params = {
            "symbol": ticker.upper(),
            "metric": "all",
            "token": self.finnhub_token
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            metric = data.get("metric", {})
            
            if metric:
                roe = metric.get("roeTTM")
                if roe is not None:
                    ratios["ROE"] = float(roe) / 100.0
                    
                deuda = metric.get("totalDebt/totalEquity", metric.get("totalDebt/totalEquityAnnual", metric.get("totalDebt/totalEquityQuarterly")))
                if deuda is not None:
                    # Finnhub suele entregar este ratio en porcentaje (ej. 150.5 para 1.5x)
                    ratios["Deuda_EBITDA"] = float(deuda) / 100.0
                    
                margen_bruto = metric.get("grossMarginTTM")
                if margen_bruto is not None:
                    ratios["Margen_Bruto"] = float(margen_bruto) / 100.0
                    
                margen_neto = metric.get("netProfitMarginTTM")
                if margen_neto is not None:
                    ratios["Margen_Neto"] = float(margen_neto) / 100.0
                    
        except Exception as e:
            print(f"Error al obtener métricas de Finnhub para {ticker}: {e}")
            
        return ratios

    def calcular_valor_intrinseco_dcf(self, ticker: str) -> dict:
        """
        Calcula una estimación del Valor Intrínseco (DCF simplificado) usando datos de Finnhub.
        """
        if not self.finnhub_token:
            return {}
            
        # 1. Obtener precio actual
        precio_actual = 0.0
        try:
            res_quote = requests.get(f"{self.base_url}/quote", params={"symbol": ticker.upper(), "token": self.finnhub_token})
            if res_quote.status_code == 200:
                precio_actual = res_quote.json().get("c", 0.0)
        except Exception as e:
            print(f"Error al obtener quote en DCF para {ticker}: {e}")

        # 2. Obtener métricas para estimar crecimiento/valoración
        dcf_est = precio_actual
        try:
            res_metric = requests.get(f"{self.base_url}/stock/metric", params={"symbol": ticker.upper(), "metric": "all", "token": self.finnhub_token})
            if res_metric.status_code == 200:
                metric = res_metric.json().get("metric", {})
                eps = metric.get("epsTTM")
                pe = metric.get("peTTM")
                roe = metric.get("roeTTM")
                
                if eps and pe and eps > 0 and pe > 0:
                    # Fórmula clásica de valoración con prima de crecimiento
                    dcf_est = float(eps) * float(pe) * 1.15
                elif roe and float(roe) > 0:
                    # Estimación basada en rentabilidad sobre recursos propios
                    dcf_est = precio_actual * (1 + (float(roe) / 100.0))
                else:
                    dcf_est = precio_actual * 1.10
        except Exception as e:
            print(f"Error al obtener métricas en DCF para {ticker}: {e}")

        return {
            "dcf": round(dcf_est, 2) if dcf_est else None,
            "Stock Price": precio_actual if precio_actual else None,
            "date": str(datetime.now().date())
        }

    def obtener_precios_historicos(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Obtiene el historial de precios diarios usando la API de Tiingo.
        """
        try:
            tiingo_token = os.getenv("TIINGO_API_KEY")
            if not tiingo_token:
                print("Advertencia: TIINGO_API_KEY no está configurada.")
                return pd.DataFrame()
                
            fecha_inicio = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices?startDate={fecha_inicio}&token={tiingo_token}"
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"Advertencia: No se encontraron datos históricos de Tiingo para {ticker.upper()}.")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
                
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df = df.sort_values('Date').reset_index(drop=True)
            return df.tail(days)
            
        except Exception as e:
            print(f"Error al obtener precios con Tiingo: {e}")
            return pd.DataFrame()

    def identificar_soportes(self, df: pd.DataFrame, window: int = 20) -> list:
        """
        Identifica niveles de soporte basados en mínimos locales.
        """
        if df.empty or len(df) < window:
            return []
            
        lows = df["Low"].values
        soportes = []
        
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window : i+window]):
                soportes.append(round(float(lows[i]), 2))
        
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
        
        precio_actual = 0.0
        if self.finnhub_token:
            endpoint = f"{self.base_url}/quote"
            params = {
                "symbol": ticker.upper(),
                "token": self.finnhub_token
            }
            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                precio_actual = float(data.get('c', 0.0))
            except Exception as e:
                print(f"Error al obtener precio actual de Finnhub para {ticker}: {e}")
                
        if precio_actual == 0.0 and not df.empty:
            precio_actual = float(df["Close"].iloc[-1])

        rsi_mensaje = "Neutral"
        if rsi is not None:
            if rsi < 30:
                rsi_mensaje = "Sobreventa / Oportunidad"
            elif rsi > 70:
                rsi_mensaje = "Sobrecompra / Riesgo"

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
        Obtiene las últimas noticias usando la API de Finnhub.io.
        """
        if not self.finnhub_token:
            return []
            
        try:
            fecha_fin = datetime.now().strftime('%Y-%m-%d')
            fecha_inicio = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
            url = f"{self.base_url}/company-news?symbol={ticker.upper()}&from={fecha_inicio}&to={fecha_fin}&token={self.finnhub_token}"
            
            response = requests.get(url)
            response.raise_for_status()
            news = response.json()
            
            if not news:
                return []
                
            textos = [f"Título: {n.get('headline')}\nFuente: {n.get('source')}\nResumen: {n.get('summary')}" for n in news[:limit]]
            return textos
            
        except Exception as e:
            print(f"Error al obtener noticias con Finnhub: {e}")
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
