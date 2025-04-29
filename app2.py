import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import StringIO

st.set_page_config(page_title="Airline Ticket Sales Forecast", layout="wide")
st.title("Airline Ticket Sales Forecasting App")

st.markdown("""
This app predicts **airline ticket sales** using the **Prophet** forecasting model. 
Upload a CSV file containing historical data to get started.

**Required Columns:**
- `Year`
- `quarter`
- `passengers`
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Combine Year and Quarter to form a Date column
    def get_quarter_start_month(q):
        return {1: 1, 2: 4, 3: 7, 4: 10}.get(int(q), 1)

    df['Month'] = df['quarter'].apply(get_quarter_start_month)
    df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=1))

    # Keep only necessary columns
    ts_df = df[['Date', 'passengers']].dropna()
    ts_df = ts_df.rename(columns={'Date': 'ds', 'passengers': 'y'})

    st.subheader("Raw Time Series Data")
    st.line_chart(ts_df.set_index('ds'))

    # Forecast using Prophet
    m = Prophet()
    m.fit(ts_df)

    future = m.make_future_dataframe(periods=8, freq='Q')
    forecast = m.predict(future)

    st.subheader("Forecast Plot")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("Forecast Data Table")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8))

else:
    st.info("Awaiting CSV file upload...")
