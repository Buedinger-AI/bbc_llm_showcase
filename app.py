import streamlit as st

# Titel der App
st.title('Hello World')



# Button, um eine Aktion auszulösen
if st.button('Say Hi'):
    st.write(f'Hi!')
    