import pandas as pd
import streamlit as st
df = pd.read_csv('monitoreo_ambiental.csv')
st.title('Actividad Clase 2')
st.table(df.describe())