
import streamlit as st
import pip as pip
import sys

pip.main(["install", "seaborn"])
pip.main(["install", "openpyxl"])
pip.main(["install", "matplotlib"])
pip.main(["install", " yfinance"])

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import yfinance as yf
import pytz
from datetime import date, timedelta
import numpy as np
import requests
import json
import pandas as pd

zonahoraria = pytz.timezone('America/Argentina/Buenos_Aires')

st.title("4 indicadores para Acciones")
index_symbol = "QQQ"

fecha_actual = date.today()

if fecha_actual.weekday() == 5:  # 5 representa el sábado
    fecha_actual = fecha_actual - timedelta(days=1)
elif fecha_actual.weekday() == 6:  # 6 representa el domingo
    fecha_actual = fecha_actual - timedelta(days=2)
hoy = fecha_actual.strftime("%Y-%m-%d")

# desde = '2000-01-01'
desde = fecha_actual - timedelta(days=365)  # tomo un año mio

symbol = st.text_input("Ingrese el código del Ticket", index_symbol)

if st.button("Ingresar"):
    symbol = symbol.strip()
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=desde, end=hoy, auto_adjust=True)
    # data = yf.download('AAPL' , auto_adjust=True)
    # data
    st.write("La fecha de inicio del período es", desde,
             "hasta hoy ", hoy, " para el ticket", symbol, ".")
    ############################################################
    # máximo histórico
    data['maxHist'] = data.Close.cummax()
    # mínimo histórico
    data['minHist'] = data.Close.cummin()
    # retono logaritmico promedio
    retorno = np.log((data.Close/data.Close.shift(1)))
    retorno = retorno.mean()

    ############################################################
    COSTO = 0

    # retorno lineal... asimetrico
    data['va'] = data.Close.pct_change() - COSTO
    # retorno logaritmico...mas simetrio
    data['va_log'] = np.log(data.Close / data.Close.shift()) - COSTO

    ############################################################
    #           cruce de media
    #
    # OJOOOO  puede ser cualquier valor... luego ver analisis de sensibilidad...mas adelante
    fast, slow = 10, 30
    # esto es una media movil exponencial
    data['sma_fast'] = data.Close.ewm(span=fast).mean()
    data['sma_slow'] = data.Close.ewm(span=slow).mean()
    data['sma_100'] = data.Close.ewm(span=100).mean()

    # Atenti, en esta linea -- Importantisimo el shift()!!
    # lo hago en funcion de ayer x eso el shift
    data['cruce'] = (data.sma_fast / data.sma_slow - 1).shift()

    data['estado'] = np.where(data.cruce > 0, 'in', 'out')

    data['accion'] = np.where((data.estado == 'in') & (data.estado.shift() == 'out'), 'Comprar',
                              np.where((data.estado == 'out') & (data.estado.shift() == 'in'), 'Vender', 'Sin Accion'))

    # nuevo DataFrame llamado compras, que contiene solo las filas de data donde la columna "accion" tiene el valor "Comprar".
    compras = data.loc[data.accion == 'Comprar']
    # crea un DataFrame llamado ventas, que contiene solo las filas de data donde la columna "accion" tiene el valor "Vender"
    ventas = data.loc[data.accion == 'Vender']

    data['cmedia'] = np.where((data.estado == 'in') & (data.accion == 'Sin Accion'), 'Tendencia a Comprar',
                              np.where((data.estado == 'out') & (data.accion == 'Sin Accion'), 'Tendencia a Vender', data['accion']))

    ######################################################################
    #                    RSI

    change = data["Close"].diff()
    change.dropna(inplace=True)

    # Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()

    #
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0

    # Verify that we did not make any mistakes
    change.equals(change_up+change_down)

    # Calculate the rolling average of average up and average down
    avg_up = change_up.rolling(14).mean()
    avg_down = change_down.rolling(14).mean().abs()

    rsi = 100 * avg_up / (avg_up + avg_down)

    data['rsi'] = rsi
    data['accionRSI'] = np.where(data.rsi >= 70, 'Tendencia para Vender', np.where(
        data.rsi <= 30, 'Tendencia para Comprar', 'Sin Accion'))

    RSIventa = data.rsi >= 70
    RSIcompra = data.rsi <= 30

    ######################################################################
    #                   MACD
    # Parámetros
    s = 12
    l = 26
    signal = 9

    data["macd"] = data.Close.ewm(span=s, min_periods=1).mean(
    ) - data.Close.ewm(span=l, min_periods=1).mean()
    data["signal"] = data.macd.ewm(span=signal, min_periods=1).mean()
    data["diff"] = data["macd"] - data["signal"]

    mask_verde = (data['diff'].shift() < 0) & (data['diff'] > 0)
    mask_roja = (data['diff'].shift() > 0) & (data['diff'] < 0)

    data.loc[mask_verde, 'MACD'] = 'Comprar'
    data.loc[mask_roja, 'MACD'] = 'Vender'
    data['MACD'] = np.where((data['diff'] < 0) & (data.MACD != 'Vender'), 'Tendencia a Vender', np.where(
        (data['diff'] >= 0) & (data.MACD != 'Comprar'), 'Tendencia a Comprar', data['MACD']))

    # a=(data.MACD != 'Vender').iloc[-1 :]
    # a=a.reset_index()

    ######################################################################
    #  Banda de Bollinger

    # Parámetros de las bandas de Bollinger
    periodo = 20
    desviacion = 2

    # Calcula la media móvil simple
    data['MMS'] = data['Close'].rolling(window=periodo).mean()

    # Calcula la desviación estándar
    data['std'] = data['Close'].rolling(window=periodo).std()

    # Calcula la banda superior e inferior
    data['BandaSuperior'] = data['MMS'] + (desviacion * data['std'])
    data['BandaInferior'] = data['MMS'] - (desviacion * data['std'])

    # Marcadores para las rupturas de rango
    rupturas_arriba = data[data['Close'] > data['BandaSuperior']]
    rupturas_abajo = data[data['Close'] < data['BandaInferior']]
    data['Bollinger'] = np.where(data['Close'] > data['BandaSuperior'], 'Vender', np.where(
        data['Close'] < data['BandaInferior'], 'Comprar', "Sin acción"))

    # DIBUJOS
    #
    #########################################################
    #   MACD
    xdate = [x.date() for x in data.index]
    plt.figure(figsize=(15, 10))

    # plot the original closing line
    plt.subplot(211)
    plt.title("Cierre del Ticket")
    plt.plot(xdate, data.maxHist, label="Máximo")
    plt.plot(xdate, data.minHist, label="Mínimo")
    plt.plot(xdate, data.Close, label="Cierre")
    plt.xlim(xdate[0], xdate[-1])
    plt.legend()
    plt.grid()

    # plot MACD and signal
    plt.subplot(212)
    plt.title("MACD")
    plt.plot(xdate, data.macd, label="macd")
    plt.plot(xdate, data.signal, label="signal")
    plt.xlim(xdate[0], xdate[-1])
    plt.legend()
    plt.grid(True)

    # Cross points
    for i in range(1, len(data)):
        if data.iloc[i-1]["diff"] < 0 and data.iloc[i]["diff"] > 0:
            cmacd = xdate[i]
            plt.scatter(xdate[i], data.iloc[i]["macd"],
                        marker="^", s=100, color="g", alpha=0.9)

        if data.iloc[i-1]["diff"] > 0 and data.iloc[i]["diff"] < 0:
            vmacd = xdate[i]
            plt.scatter(xdate[i], data.iloc[i]["macd"],
                        marker="v", s=100, color="r", alpha=0.9)

    st.pyplot(plt)

    #########################################################
    #  RSI

    fig, ax = plt.subplots(figsize=(15, 4))
    # Plot the RSI
    plt.title('Relative Strength Index (RSI)')
    ax.plot(data.rsi, color='orange', lw=1, label='RSI de 14')
    # Add two horizontal lines, signalling the buy and sell ranges.
    # Oversold
    ax.axhline(30, linestyle='--', linewidth=1.5, color='green')
    # Overbought
    ax.axhline(70, linestyle='--', linewidth=1.5, color='red')

    # Display the charts
    ax.legend()
    ax.grid(axis='both')
    st.pyplot(fig)

    #########################################################
    #  Bandas Bollinger

    fig, ax = plt.subplots(figsize=(15, 4))
    plt.title("Bandas de Bollinger")
    ax.plot(data.Close, '-k', lw=1, label='Precio')
    ax.plot(data.BandaSuperior, '--g', lw=1, label='Banda Superior')
    ax.plot(data.BandaInferior, '--r', lw=1, label='Banda Inferior')
    ax.fill_between(data.index, data['BandaInferior'],
                    data['BandaSuperior'], facecolor='orange', alpha=0.1)
    ax.plot(rupturas_arriba.Close, marker='v', lw=0,
            markersize=10, color='r', label='Vender')
    ax.plot(rupturas_abajo.Close, marker='^', lw=0,
            markersize=10, color='g', label='Comprar')
    ax.legend()
    ax.grid(axis='both')
    st.pyplot(fig)

    #########################################################################
    #     CRUCE DE MEDIAS

    fig, ax = plt.subplots(figsize=(15, 4))
    # lw: linewidth..
    plt.title("Cruce de Medias")
    ax.plot(data.Close, '-k', lw=1, label='Precio')
    ax.plot(data.sma_fast, '--g', lw=1, label='SMA Rapida (10)')
    ax.plot(data.sma_slow, '--r', lw=1, label='SMA Lenta (30)')
    ax.plot(data.sma_100, '--b', lw=1, label='SMA Lenta (100)')

    ax.plot(compras.Close, marker='^', lw=0,
            markersize=10, color='g', label='Compras')
    ax.plot(ventas.Close, marker='v', lw=0,
            markersize=10, color='r', label='Ventas')
    ax.legend()
    ax.grid(axis='both')
    st.pyplot(fig)

    #########################################################################
    #      DATOS

    st.write("MACD: último día que dio Comprar: ", cmacd.strftime("%Y-%m-%d"))
    st.write("MACD: último día que dio Vender: ", vmacd.strftime("%Y-%m-%d"))

    # st.write("RSI: último día que dio Comprar: ", cmacd.strftime("%Y-%m-%d"))
    # st.write("RSI: último día que dio Vender: ", vmacd.strftime("%Y-%m-%d"))

    st.write("Bolling: último día que dio Comprar: ",
             rupturas_abajo.index[-1].strftime("%Y-%m-%d"))
    st.write("Bollinger: último día que dio Vender: ",
             rupturas_arriba.index[-1].strftime("%Y-%m-%d"))

    st.write("C.Medias: último día que dio Comprar: ",
             compras.index[-1].strftime("%Y-%m-%d"))
    st.write("C.Medias: último día que dio Vender: ",
             ventas.index[-1].strftime("%Y-%m-%d"))

    st.title("Tendencia a la fecha")
    st.write(data[['MACD', 'accionRSI', 'Bollinger', 'cmedia',
             'maxHist', 'minHist', 'Close']].tail(1))

    st.title("Tendencia de los últimos 10 días")
    st.write(data[['MACD', 'accionRSI', 'Bollinger', 'cmedia',
             'maxHist', 'minHist', 'Close']].tail(10))
    # columnas_seleccionadas = ['Bollinger',
    #                         'Close', 'BandaSuperior', 'BandaInferior']
    # df_seleccionado = data[columnas_seleccionadas]
    # st.write(df_seleccionado)

    nombre_archivo = symbol+'.xlsx'
    st.write(nombre_archivo)
    # Guardar el DataFrame en Excel
    data.to_excel(nombre_archivo, index=False)
