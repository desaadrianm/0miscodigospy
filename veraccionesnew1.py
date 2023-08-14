import streamlit as st
import pip as pip

pip.main(["install", "yfinance"])
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta
import numpy as np
import pandas as pd

def calcular_adx(datos, periodo=14):
    datos['High-Low'] = datos['High'] - datos['Low']
    datos['High-PrevClose'] = abs(datos['High'] - datos['Close'].shift(1))
    datos['Low-PrevClose'] = abs(datos['Low'] - datos['Close'].shift(1))

    datos['TR'] = datos[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    datos['+DM'] = (datos['High'] - datos['High'].shift(1)).where(
        (datos['High'] - datos['High'].shift(1)) > (datos['Low'].shift(1) - datos['Low']),
        0
    )
    datos['-DM'] = (datos['Low'].shift(1) - datos['Low']).where(
        (datos['Low'].shift(1) - datos['Low']) > (datos['High'] - datos['High'].shift(1)),
        0
    )

    datos['TR_EMA'] = datos['TR'].ewm(span=periodo, adjust=False).mean()
    datos['+DM_EMA'] = datos['+DM'].ewm(span=periodo, adjust=False).mean()
    datos['-DM_EMA'] = datos['-DM'].ewm(span=periodo, adjust=False).mean()

    datos['+DI'] = (100 * datos['+DM_EMA'] / datos['TR_EMA']).fillna(0)
    datos['-DI'] = (100 * datos['-DM_EMA'] / datos['TR_EMA']).fillna(0)

    datos['DX'] = (100 * abs(datos['+DI'] - datos['-DI']) / (datos['+DI'] + datos['-DI'])).fillna(0)
    datos['ADX'] = datos['DX'].ewm(span=periodo, adjust=False).mean()
    datos['dif'] = datos['+DI'] - datos['-DI']
    datos['accion'] = np.where((datos['dif'] > 0) & (datos['dif'].shift() < 0) ,"sube",
                               np.where( (datos['dif'] < 0) & (datos['dif'].shift() > 0), 'baja', 'Sin Accion'))
    return datos[['Close','+DI', '-DI', 'ADX','dif', 'accion']]

index_symbol = ""
clave1 = st.text_input("Ingrese Clave", index_symbol)
if  clave1 == "1963":
        
    fecha_actual = date.today()

    if fecha_actual.weekday() == 5:  # 5 representa el sábado
        fecha_actual = fecha_actual - timedelta(days=1)
    elif fecha_actual.weekday() == 6:  # 6 representa el domingo
        fecha_actual = fecha_actual - timedelta(days=2)
    hoy = fecha_actual.strftime("%Y-%m-%d")

    # desde = '2000-01-01'
    desde = fecha_actual - timedelta(days=180)  # tomo dos año mio

    st.title("Indicadores Técnico para Acciones")
    index_symbol = ""
    symbol = st.text_input("Ingrese el código del Ticket", index_symbol)
    if st.button("Ingresar"):
        symbol = symbol.strip()
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=desde, end=hoy, auto_adjust=True)

        if not data.empty:

            st.write("La fecha de inicio del período es", desde,
                    "hasta hoy ", hoy, " para el ticket", symbol, ".")
            ############################################################
            # máximo histórico
            data['maxHist'] = data.Close.cummax()
            # mínimo histórico
            data['minHist'] = data.Close.cummin()

            #########################################################################
            # Llamada a la función para calcular el ADX

            resultado_adx = calcular_adx(data)

            compras = resultado_adx.loc[resultado_adx.accion=='sube'] # nuevo DataFrame llamado compras, que contiene solo las filas de df donde la columna "accion" tiene el valor "Comprar".
            ventas = resultado_adx.loc[resultado_adx.accion=='baja'] # crea un DataFrame llamado ventas, que contiene solo las filas de df donde la columna "accion" tiene el valor "Vender"

            plt.style.use("seaborn-v0_8-whitegrid")
            fig , ax = plt.subplots(figsize=(20,10))
            plt.plot(data.index, resultado_adx['Close'],'-', lw=2, label='Close', color='black')
            plt.grid(True)
            st.pyplot(plt)

            plt.figure(figsize=(20, 10))
            #plt.plot(data.index, resultado_adx['ADX'],'-', lw=2, label='ADX', color='blue')
            plt.plot(data.index, resultado_adx['+DI'],'-', lw=2, label='+DI', color='green')
            plt.plot(data.index, resultado_adx['-DI'],'-', lw=2, label='-DI', color='red')

            # Señales de compra (Cuando +DI cruza por encima de -DI y ADX es mayor que cierto umbral)
            plt.plot(compras['+DI'], marker='^', lw=0, markersize=10, color='g', label='Compras')
            plt.plot(ventas['+DI'], marker='v', lw=0, markersize=10, color='r', label='Ventas')

            plt.legend()
            plt.title('Gráfico DMI (ADX)')
            plt.xlabel('Fecha')
            plt.ylabel('DI')
            plt.grid(True)

            st.pyplot(plt)