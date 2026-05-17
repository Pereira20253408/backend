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
            
        ratios["salud_score"] = self.calcular_puntaje_salud(ratios)
        return ratios

    def calcular_puntaje_salud(self, ratios: dict) -> int:
        """
        Calcula un puntaje de salud financiera (0 a 100) basado en ROE, Deuda/Equity y Márgenes.
        """
        score = 0
        
        # 1. ROE (máx 25 pts)
        roe = ratios.get("ROE")
        if roe is not None:
            if roe >= 0.20: score += 25
            elif roe >= 0.15: score += 20
            elif roe >= 0.10: score += 15
            elif roe > 0: score += 10
            
        # 2. Deuda/Equity (o Deuda_EBITDA) (máx 25 pts)
        deuda = ratios.get("Deuda_EBITDA")
        if deuda is not None:
            if deuda < 1.0: score += 25
            elif deuda < 2.0: score += 20
            elif deuda < 3.0: score += 15
            elif deuda < 4.0: score += 10
            
        # 3. Margen Bruto (máx 25 pts)
        margen_bruto = ratios.get("Margen_Bruto")
        if margen_bruto is not None:
            if margen_bruto >= 0.40: score += 25
            elif margen_bruto >= 0.30: score += 20
            elif margen_bruto >= 0.20: score += 15
            elif margen_bruto > 0: score += 10
            
        # 4. Margen Neto (máx 25 pts)
        margen_neto = ratios.get("Margen_Neto")
        if margen_neto is not None:
            if margen_neto >= 0.15: score += 25
            elif margen_neto >= 0.10: score += 20
            elif margen_neto >= 0.05: score += 15
            elif margen_neto > 0: score += 10
            
        if score == 0 and any(v is not None for v in ratios.values()):
            score = 50
        elif score == 0:
            score = 65
            
        return min(score, 100)

    def calcular_valor_intrinseco_dcf(self, ticker: str) -> dict:
        """
        Calcula una estimación del Valor Intrínseco (DCF simplificado) usando datos de Finnhub.
        """
        if not self.finnhub_token:
            return {
                "wacc_default": 9.5,
                "growth_default": 10.0
            }
            
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
        eps = None
        pe = None
        roe = None
        market_cap = None
        acciones_circulacion = 1000.0 # en millones
        flujo_caja = 5000.0 # en millones
        deuda_neta = 2000.0 # en millones
        
        wacc_default = 9.5
        growth_default = 10.0

        try:
            res_metric = requests.get(f"{self.base_url}/stock/metric", params={"symbol": ticker.upper(), "metric": "all", "token": self.finnhub_token})
            if res_metric.status_code == 200:
                metric = res_metric.json().get("metric", {})
                eps = metric.get("epsTTM")
                pe = metric.get("peTTM")
                roe = metric.get("roeTTM")
                market_cap = metric.get("marketCapitalization")
                
                # --- CÁLCULO DE WACC REAL Y PRECISO (PONDERADO) ---
                beta = metric.get("beta")
                try:
                    beta = float(beta) if beta is not None else 1.0
                except (ValueError, TypeError):
                    beta = 1.0

                # 1. Costo del Capital Propio (Cost of Equity usando CAPM)
                rf = 4.3   
                erp = 5.2  
                cost_of_equity = rf + (beta * erp)

                # 2. Extraer Capital (Equity)
                try:
                    E = float(market_cap) if market_cap else 0.0
                except (ValueError, TypeError):
                    E = 0.0

                # 3. Extraer Deuda (Con ingeniería inversa si la API la oculta)
                deuda_cruda = metric.get("totalDebtAnnual", metric.get("totalDebtQuarterly", metric.get("netDebtAnnual")))
                D = float(deuda_cruda) if deuda_cruda else 0.0
                
                if D == 0.0 and E > 0:
                    # Buscamos el ratio Deuda/Capital que Finnhub sí entrega gratis
                    ratio_de = metric.get("totalDebt/totalEquity", metric.get("totalDebt/totalEquityAnnual"))
                    if ratio_de:
                        # Finnhub entrega esto como porcentaje (ej. 150 para 1.5)
                        D = E * (float(ratio_de) / 100.0)
                    else:
                        # Último recurso salvavidas: Promedio de deuda corporativa (15%)
                        D = E * 0.15
                
                V = E + D  

                # 4. Costo de la Deuda y Tasa Impositiva
                cost_of_debt = rf + 2.0  
                tax_rate = 0.21          

                # 5. Fórmula del WACC
                if V > 0 and E > 0:
                    peso_equity = E / V
                    peso_debt = D / V
                    wacc_calculado = (peso_equity * cost_of_equity) + (peso_debt * cost_of_debt * (1 - tax_rate))
                else:
                    wacc_calculado = cost_of_equity

                # Límites lógicos 
                if wacc_calculado < 5.0: wacc_calculado = 5.0
                elif wacc_calculado > 18.0: wacc_calculado = 18.0

                wacc_default = round(wacc_calculado, 1)
                
                # --- DEBUG WACC ---
                # print(f"\n--- DEBUG WACC para {ticker.upper()} ---")
                # print(f"Market Cap (E): {E:.2f}")
                # print(f"Deuda (D) Calculada: {D:.2f}")
                # if V > 0:
                #     print(f"Peso Equity: {(E/V)*100:.2f}%")
                #     print(f"Peso Deuda: {(D/V)*100:.2f}%")
                # print(f"Costo de Equity: {cost_of_equity:.2f}%")
                # print(f"Costo de Deuda (neto): {(cost_of_debt * (1 - tax_rate)):.2f}%")
                # print(f"WACC Final: {wacc_calculado:.2f}%")
                # print("--------------------------------\n")
                # -------------------------------------------------

                # --- EXTRACCIÓN DE CRECIMIENTO ESTIMADO FUTURO ---
                growth = metric.get("epsGrowth3Y", metric.get("epsGrowth5Y", metric.get("revenueGrowth5Y")))
                try:
                    growth = float(growth) if growth is not None else 10.0
                    if growth < 2.0: growth = 2.0
                    elif growth > 20.0: growth = 20.0
                except (ValueError, TypeError):
                    growth = 10.0
                growth_default = round(growth, 1)

                # --- EXTRACCIÓN DE FLUJO DE CAJA (FCF) ---
                flujo_caja_total = metric.get("freeCashFlowAnnual", metric.get("netIncomeTTM"))
                try:
                    flujo_caja_total = float(flujo_caja_total) if flujo_caja_total else 0.0
                except (ValueError, TypeError):
                    flujo_caja_total = 0.0

                flujo_caja = flujo_caja_total
                deuda_neta = D

                # --- CÁLCULO REAL DEL DCF (2 ETAPAS) ---
                if flujo_caja_total > 0 and wacc_calculado > 0 and E > 0:
                    wacc_decimal = wacc_calculado / 100.0
                    growth_decimal = growth_default / 100.0
                    tasa_terminal = 0.025  # Crecimiento a perpetuidad (2.5%)
                    
                    valor_presente_flujos = 0.0
                    flujo_proyectado = flujo_caja_total
                    
                    # 1. Proyectar y descontar flujos (Años 1-5)
                    for año in range(1, 6):
                        flujo_proyectado *= (1 + growth_decimal)
                        valor_descontado = flujo_proyectado / ((1 + wacc_decimal) ** año)
                        valor_presente_flujos += valor_descontado
                    
                    # 2. Valor Terminal (Gordon Growth Model)
                    flujo_año_5 = flujo_proyectado
                    flujo_terminal = flujo_año_5 * (1 + tasa_terminal)
                    
                    if wacc_decimal > tasa_terminal:
                        valor_terminal = flujo_terminal / (wacc_decimal - tasa_terminal)
                        vp_valor_terminal = valor_terminal / ((1 + wacc_decimal) ** 5)
                    else:
                        vp_valor_terminal = 0.0
                    
                    # 3. Enterprise Value
                    enterprise_value = valor_presente_flujos + vp_valor_terminal
                    
                    # 4. Equity Value (EV - Deuda)
                    valor_intrinseco_total = enterprise_value - D
                    
                    # 5. Valor por Acción
                    acciones_circulacion = E / precio_actual if precio_actual > 0 else 1.0
                    if acciones_circulacion > 0:
                        dcf_est = valor_intrinseco_total / acciones_circulacion
                    else:
                        dcf_est = None
                else:
                    dcf_est = None
                    
                if dcf_est is not None and dcf_est < 0:
                    dcf_est = None

        except Exception as e:
            print(f"Error al obtener métricas en DCF para {ticker}: {e}")

        return {
            "dcf": round(dcf_est, 2) if dcf_est else None,
            "Stock Price": precio_actual if precio_actual else None,
            "date": str(datetime.now().date()),
            "datos_crudos": {
                "flujo_caja": round(flujo_caja, 2),
                "deuda_neta": round(deuda_neta, 2),
                "acciones_circulacion": round(acciones_circulacion, 2)
            },
            "wacc_default": wacc_default,
            "growth_default": growth_default
        }

    def obtener_precios_objetivo(self, ticker: str, precio_actual: float = 0.0) -> dict:
        """
        Consulta el endpoint de Finnhub de precios objetivo (/stock/price-target).
        Retorna un diccionario con los targets alto, moderado y bajo.
        """
        targets = {
            "alto": None,
            "moderado": None,
            "bajo": None
        }
        if self.finnhub_token:
            endpoint = f"{self.base_url}/stock/price-target"
            params = {
                "symbol": ticker.upper(),
                "token": self.finnhub_token
            }
            try:
                response = requests.get(endpoint, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        targets["alto"] = data.get("targetHigh")
                        targets["moderado"] = data.get("targetMean")
                        targets["bajo"] = data.get("targetLow")
            except Exception as e:
                print(f"Error al obtener price-target de Finnhub para {ticker}: {e}")
        
        # Fallback si no hay datos de analistas pero tenemos precio actual
        if (targets["moderado"] is None or targets["moderado"] == 0) and precio_actual > 0:
            targets["alto"] = round(precio_actual * 1.25, 2)
            targets["moderado"] = round(precio_actual * 1.10, 2)
            targets["bajo"] = round(precio_actual * 0.85, 2)
            
        return targets

    def obtener_precios_historicos(self, ticker: str, periodo: str = '1y') -> pd.DataFrame:
        """
        Obtiene el historial de precios diarios usando la API de Tiingo.
        """
        try:
            tiingo_token = os.getenv("TIINGO_API_KEY")
            if not tiingo_token:
                print("Advertencia: TIINGO_API_KEY no está configurada.")
                return pd.DataFrame()
                
            dias = 365
            if periodo == '1y':
                dias = 365
            elif periodo == '2y':
                dias = 730
            elif periodo == '3y':
                dias = 1095
            elif periodo == '5y':
                dias = 1825
            elif periodo == '10y':
                dias = 3650
                
            fecha_inicio = (datetime.now() - timedelta(days=dias)).strftime('%Y-%m-%d')
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
            return df.tail(dias)
            
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

    def obtener_analisis_tecnico(self, ticker: str, periodo: str = '1y') -> dict:
        """
        Realiza un análisis técnico simplificado: RSI y Soportes.
        """
        df = self.obtener_precios_historicos(ticker, periodo=periodo)
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

        historico = []
        if not df.empty:
            for _, row in df.iterrows():
                fecha_str = str(row['Date']).split('T')[0].split(' ')[0]
                historico.append({
                    "date": fecha_str,
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2)
                })

        return {
            "rsi": rsi,
            "rsi_mensaje": rsi_mensaje,
            "soportes": soportes,
            "soporte_cercano": soporte_cercano,
            "distancia_soporte_porcentaje": distancia_soporte,
            "precio_actual": precio_actual,
            "historico": historico
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

    def chatear_ia(self, ticker: str, mensaje: str, historial: list = []) -> dict:
        """
        Chatbot conversacional libre sobre un ticker, inyectando datos fundamentales y técnicos.
        """
        if not self.model:
            return {"respuesta": "IA no configurada (falta GOOGLE_API_KEY en el backend)."}
            
        ratios = self.obtener_ratios_salud(ticker)
        dcf_data = self.calcular_valor_intrinseco_dcf(ticker)
        tecnico = self.obtener_analisis_tecnico(ticker)

        historial_str = ""
        if historial:
            historial_str = "\n".join([f"{h.get('role', 'Usuario')}: {h.get('content', '')}" for h in historial])

        contexto = f"""
        Estás actuando como un Asistente Financiero Experto de la plataforma Quantix.
        Aquí tienes los datos actuales en tiempo real del ticker {ticker.upper()}:
        
        - Precio Actual: ${dcf_data.get('Stock Price') or tecnico.get('precio_actual', 'N/A')}
        - Valor Intrínseco Estimado (DCF): ${dcf_data.get('dcf', 'N/A')}
        - ROE: {ratios.get('ROE', 'N/A')}
        - Margen Bruto: {ratios.get('Margen_Bruto', 'N/A')}
        - Margen Neto: {ratios.get('Margen_Neto', 'N/A')}
        - Deuda/EBITDA: {ratios.get('Deuda_EBITDA', 'N/A')}
        - RSI (14 días): {tecnico.get('rsi', 'N/A')} ({tecnico.get('rsi_mensaje', 'N/A')})
        - Soporte Técnico Cercano: ${tecnico.get('soporte_cercano', 'N/A')}
        
        Historial de la conversación previa:
        {historial_str}
        
        Pregunta del usuario: {mensaje}
        
        Responde de forma profesional, clara, concisa y directa a la pregunta del usuario utilizando este contexto financiero y tu conocimiento general sobre {ticker.upper()}.
        """

        try:
            response = self.model.generate_content(contexto)
            return {"respuesta": response.text}
        except Exception as e:
            print(f"Error en chatear_ia: {e}")
            return {"respuesta": "Lo siento, ha ocurrido un error al generar la respuesta de la IA."}

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
