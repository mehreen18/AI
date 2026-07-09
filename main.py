
import streamlit as st

# Title jo page ke upar dikhega
st.title("Mera Pehla Streamlit App")

st.write("Ye ek simple practice app hai taake samajh aaye Streamlit kaise kaam karta hai.")

# User se input lena
naam = st.text_input("Apna naam likhein:")
umar = st.slider("Apni age select karein:", 1, 100, 25)

# Button
if st.button("Salam Bolo"):
    st.write(f"Assalam-o-Alaikum, {naam}! Aapki age {umar} saal hai.")

# Ek simple calculation ka example
number1 = st.number_input("Pehla number:")
number2 = st.number_input("Doosra number:")

if st.button("Add Karo"):
    result = number1 + number2
    st.success(f"Result: {result}")