import streamlit as st

def main():
    st.title("Data Analytics Project - World Cup 2022 Prediction")
    menu = ["Squad Strength", "EDA and Data Processing", "Model Creation", "Predicting the World Cup"]
    choice = st.sidebar.selectbox("Menu", menu)



if __name__ == '__main__':
    main()
