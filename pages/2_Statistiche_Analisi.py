import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

def header_block(title, subtitle, description, image, category):
    st.markdown(f"### {category}")
    st.title(title)
    st.markdown(description)
    st.markdown("---")

def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# Recupero o inizializzazione del dataframe in st.session_state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(columns=[
        'Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 
        'RPE', 'Ore Sonno', 'Stress Lavoro', 'Rischio Infortunio'
    ])

df = st.session_state['df']

st.markdown("## Modulo — Analisi Stato di Formazione e Storico Allenamenti")
st.markdown("Compila i dati della nuova sessione per aggiornare lo stato di forma e visualizzare la tabella riepilogativa.")

# Form per la compilazione/aggiunta di un nuovo allenamento
with st.form("form_nuovo_allenamento"):
    st.subheader("Registra Nuova Sessione")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        giorno_input = st.date_input("Data Allenamento")
        distanza_input = st.number_input("Distanza (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        velocita_input = st.number_input("Velocità Media (km/h)", min_value=0.0, max_value=30.0, value=11.5, step=0.1)
        
    with col_f2:
        fc_input = st.number_input("FC Media (bpm)", min_value=60, max_value=220, value=150)
        rpe_input = st.slider("Sforzo Percepito (RPE 1-10)", min_value=1, max_value=10, value=5)
        ore_sonno = st.number_input("Ore di Sonno", min_value=0.0, max_value=14.0, value=7.5, step=0.5)
        
    with col_f3:
        stress_lavoro = st.slider("Stress Lavorativo (1-10)", min_value=1, max_value=10, value=3)
        
    submit_btn = st.form_submit_button("Aggiungi Allenamento e Aggiorna Stato")
    
    if submit_btn:
        # Calcolo euristico automatico del rischio infortunio basato su carico e recupero
        rischio_calc = 1 if (rpe_input >= 8 or ore_sonno < 6.0 or (stress_lavoro >= 7 and rpe_input >= 6)) else 0
        
        nuova_riga = pd.DataFrame([{
            'Giorno': pd.to_datetime(giorno_input),
            'Distanza (km)': distanza_input,
            'Velocità (km/h)': velocita_input,
            'FC Media': fc_input,
            'RPE': rpe_input,
            'Ore Sonno': ore_sonno,
            'Stress Lavoro': stress_lavoro,
            'Rischio Infortunio': rischio_calc
        }])
        
        st.session_state['df'] = pd.concat([st.session_state['df'], nuova_riga], ignore_index=True)
        df = st.session_state['df']
        st.success("Sessione registrata con successo e stato di forma aggiornato!")

st.markdown("---")
st.subheader("Panoramica Stato di Forma")

if df.empty:
    st.info("Nessun allenamento registrato. Compila il modulo sopra per iniziare l'analisi.")
else:
    # KPI di sintesi dello stato di forma
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("KM Totali", f"{df['Distanza (km)'].sum():.1f} km")
    col_m2.metric("Sessioni Totali", f"{len(df)}")
    col_m3.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f} ore")
    col_m4.metric("Giorni a Rischio", f"{df['Rischio Infortunio'].sum()}")

    st.markdown("---")
    st.subheader("Tabella Storico Allenamenti")

    # Preparazione tabella pulita per la visualizzazione
    tab_data = df.copy()
    tab_data['Giorno'] = pd.to_datetime(tab_data['Giorno']).dt.strftime('%d/%m/%Y')
    tab_data['Stato Rischio'] = tab_data['Rischio Infortunio'].apply(lambda x: 'ALTO' if x == 1 else 'OK')
    
    colonne_visibili = ['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro', 'Stato Rischio']
    tab_display = tab_data[colonne_visibili].sort_values(by='Giorno', ascending=False)

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=list(tab_display.columns), 
            fill_color='#111827', 
            align='center', 
            font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")
        ),
        cells=dict(
            values=[tab_display[col] for col in tab_display.columns], 
            fill_color='#0E1420', 
            align='center', 
            font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), 
            height=30
        )
    )])
    fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
    st.plotly_chart(style_fig(fig_table), use_container_width=True)
