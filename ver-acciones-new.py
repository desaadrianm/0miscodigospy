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

    return datos[['+DI', '-DI', 'ADX']]

def calcular_estocastico(datos, periodo=14, suavizado=3):
    datos['Min_Low'] = datos['Low'].rolling(window=periodo, min_periods=1).min()
    datos['Max_High'] = datos['High'].rolling(window=periodo, min_periods=1).max()

    datos['%K'] = 100 * (datos['Close'] - datos['Min_Low']) / (datos['Max_High'] - datos['Min_Low'])
    datos['%D'] = datos['%K'].rolling(window=suavizado, min_periods=1).mean()

    return datos[['%K', '%D']]

fecha_actual = date.today()

if fecha_actual.weekday() == 5:  # 5 representa el sábado
    fecha_actual = fecha_actual - timedelta(days=1)
elif fecha_actual.weekday() == 6:  # 6 representa el domingo
    fecha_actual = fecha_actual - timedelta(days=2)
hoy = fecha_actual.strftime("%Y-%m-%d")

# desde = '2000-01-01'
desde = fecha_actual - timedelta(days=730)  # tomo dos año mio

st.title("6 indicadores para Acciones")
index_symbol = "QQQ"
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

    #plt.show()
    st.pyplot(plt)
    #########################################################
    #  RSI

    fig, ax = plt.subplots(figsize=(15, 4))

    # Plot the RSI
    plt.title('Relative Strength Index (RSI)')
    plt.plot(data.rsi, color='orange', lw=1, label='RSI de 14')

    # Add two horizontal lines, signalling the buy and sell ranges.
    # Oversold
    plt.axhline(30, linestyle='--', linewidth=1.5, color='green')
    # Overbought
    plt.axhline(70, linestyle='--', linewidth=1.5, color='red')

    # Add points of compra and venta
    plt.scatter(data.index[RSIventa], data.rsi[RSIventa], color='red', marker='v', label='Venta')
    plt.scatter(data.index[RSIcompra], data.rsi[RSIcompra], color='green', marker='^', label='Compra')

    # Display the charts
    plt.legend()
    plt.grid(axis='both')
    #plt.show()
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
    #plt.show()
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
    plt.show()
    st.pyplot(fig)

    #########################################################################
    # Llamada a la función para calcular el ADX

    resultado_adx = calcular_adx(data)
    # Crear un gráfico para visualizar el ADX junto con señales de compra y venta
    plt.figure(figsize=(15, 4))

    # Gráfico del ADX
    plt.plot(data.index, resultado_adx['ADX'], label='ADX', color='blue')

    # Señales de compra (Cuando +DI cruza por encima de -DI y ADX es mayor que cierto umbral)
    plt.scatter(resultado_adx.index[resultado_adx['+DI'] > resultado_adx['-DI']],
                resultado_adx['ADX'][resultado_adx['+DI'] > resultado_adx['-DI']],
                marker='^', color='green', label='Señal de compra', s=100)

    # Señales de venta (Cuando -DI cruza por encima de +DI y ADX es mayor que cierto umbral)
    plt.scatter(resultado_adx.index[resultado_adx['-DI'] > resultado_adx['+DI']],
                resultado_adx['ADX'][resultado_adx['-DI'] > resultado_adx['+DI']],
                marker='v', color='red', label='Señal de venta', s=100)

    plt.legend()
    plt.title('Gráfico del ADX con señales de compra y venta')
    plt.xlabel('Fecha')
    plt.ylabel('ADX')
    plt.grid(True)
    #plt.show()
    st.pyplot(plt)

    #########################################################################
    #
    #            estocástico
    # Llamada a la función para calcular el indicador estocástico
    resultado_estocastico = calcular_estocastico(data)

    # Crear un gráfico para visualizar el indicador estocástico junto con señales de compra y venta
    plt.figure(figsize=(15, 4))

    # Gráfico del %K y %D
    plt.plot(data.index, resultado_estocastico['%K'], label='%K', color='blue')
    plt.plot(data.index, resultado_estocastico['%D'], label='%D', color='red')

    # Definir los umbrales de sobrecompra y sobreventa (por ejemplo, %K > 80 para sobrecompra y %K < 20 para sobreventa)
    umbral_sobrecompra = 80
    umbral_sobreventa = 20

    # Señales de compra (Cuando %K cruza por debajo del umbral de sobreventa y %D está por debajo del umbral)
    plt.scatter(resultado_estocastico.index[(resultado_estocastico['%K'] < umbral_sobreventa) & (resultado_estocastico['%D'] < umbral_sobreventa)],
                resultado_estocastico['%K'][(resultado_estocastico['%K'] < umbral_sobreventa) & (resultado_estocastico['%D'] < umbral_sobreventa)],
                marker='^', color='green', label='Señal de compra', s=100)

    # Señales de venta (Cuando %K cruza por encima del umbral de sobrecompra y %D está por encima del umbral)
    plt.scatter(resultado_estocastico.index[(resultado_estocastico['%K'] > umbral_sobrecompra) & (resultado_estocastico['%D'] > umbral_sobrecompra)],
                resultado_estocastico['%K'][(resultado_estocastico['%K'] > umbral_sobrecompra) & (resultado_estocastico['%D'] > umbral_sobrecompra)],
                marker='v', color='red', label='Señal de venta', s=100)

    plt.axhline(y=umbral_sobreventa, color='gray', linestyle='dashed', label='Sobreventa')
    plt.axhline(y=umbral_sobrecompra, color='gray', linestyle='dashed', label='Sobrecompra')

    plt.legend()
    plt.title('Gráfico del indicador estocástico con señales de compra y venta')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.grid(True)
    #plt.show()
    st.pyplot(plt)

    #########################################################################
    #     CRUCE DE MEDIAS
    st.write("MACD: último día que dio Comprar: ", cmacd.strftime("%Y-%m-%d"))
    st.write("MACD: último día que dio Vender: " ,vmacd.strftime("%Y-%m-%d"))
    st.write("-------------------------------------------")
    filas_true = RSIventa[RSIcompra == True]
    st.write("RSI: último día que dio Comprar: ", filas_true.index[-1].date())
    filas_true1 = RSIventa[RSIventa == True]
    st.write("RSI: último día que dio Vender: " , filas_true1.index[-1].date())
    st.write("-------------------------------------------")
    st.write("Bolling: último día que dio Comprar: ", rupturas_arriba.index[-1].strftime("%Y-%m-%d"))
    st.write("Bollinger: último día que dio Vender: ", rupturas_abajo.index[-1].strftime("%Y-%m-%d"))
    st.write("-------------------------------------------")
    st.write("C.Medias: último día que dio Comprar: ", compras.index[-1].strftime("%Y-%m-%d"))
    st.write("C.Medias: último día que dio Vender: ", ventas.index[-1].strftime("%Y-%m-%d"))
    st.write("-------------------------------------------")
    condicion_compra = resultado_adx['+DI'] > resultado_adx['-DI']
    señales_compra_serie = resultado_adx[condicion_compra]['ADX']
    st.write("ADX: último día que dio Comprar: ", señales_compra_serie.index[-1].date())
    condicion_venta = resultado_adx['-DI'] > resultado_adx['+DI']
    señales_venta_serie = resultado_adx[condicion_venta]['ADX']
    st.write("ADX: último día que dio Vender: ", señales_venta_serie.index[-1].date())
    st.write("-------------------------------------------")
    # Filtrar las filas que cumplen la condición para la señal de compra
    condicion_compra = (resultado_estocastico['%K'] < umbral_sobreventa) & (resultado_estocastico['%D'] < umbral_sobreventa)
    señales_compra_serie = resultado_estocastico[condicion_compra]['%K']
    st.write("ESTOCASTICO: último día que dio Comprar: ", señales_compra_serie.index[-1].date())
    # Filtrar las filas que cumplen la condición para la señal de venta
    condicion_venta = (resultado_estocastico['%K'] > umbral_sobrecompra) & (resultado_estocastico['%D'] > umbral_sobrecompra)
    señales_venta_serie = resultado_estocastico[condicion_venta]['%K']
    st.write("ESTOCASTICO: último día que dio Vender: ", señales_venta_serie.index[-1].date())
    st.write("-------------------------------------------")

    #########################################################################
    #     A EXCEL
    # data.reset_index(inplace=True)
    # nombre_archivo = symbol +'.xlsx'  # Puedes cambiar el nombre del archivo según prefieras
    # data.to_excel(nombre_archivo)
    # # Descargar el archivo
    # files.download(nombre_archivo)
    # st.write("DataFrame guardado exitosamente en el archivo Excel:", nombre_archivo)
else:
     st.write("Ingrese Ticket")
