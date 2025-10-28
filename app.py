import streamlit as st
from main import GrokipediaSimulator

st.title("Grokipedia Simulator")
topic = st.text_input("Topic", "Grokipedia")
if st.button("Run"):
    sim = GrokipediaSimulator()
    output = sim.run_simulation(topic)
    st.text(output)
