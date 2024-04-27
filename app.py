import streamlit as st
import logistics_model
from PIL import Image

# Preg,Plas,Pres,skin,test,mass,pedi,age,class

def main():
    im = Image.open("hosp.ico")
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=im)
    st.header("Diabetes Prediction App 🏥")
    with st.form(key='columns_in_form'):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            Preg = st.number_input("Enter Preg: ",placeholder="Pregnancy Value",step=1)
        with c2:
            plas = st.number_input("Enter Plas: ",placeholder="Pregnancy Value",step=1)
        with c3:
            pres = st.number_input("Enter Pres: ",placeholder="Pregnancy Value",step=1)
        with c4:
            skin = st.number_input("Enter skin: ",placeholder="Pregnancy Value",step=1)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            test = st.number_input("Enter test: ",placeholder="Pregnancy Value",step=1)
        with c6:
            mass = st.number_input("Enter mass: ",placeholder="Pregnancy Value")
        with c7:
            pedi = st.number_input("Enter pedi: ",placeholder="Pregnancy Value")
        with c8:
            age = st.number_input("Enter age: ",placeholder="Pregnancy Value",step=1)
        
        if st.form_submit_button("Submit Values"):
            st.subheader("Prediction")
            with st.spinner("predicting..."):
                import time
                time.sleep(1)
                result = logistics_model.model_prediction(Preg,plas,pres,skin,test,mass,pedi,age)
                st.write("Diabetes = ", result)
        # st.spinner("Predicting...")
        # pass
    #     submitButton = st.form_submit_button(label = 'Calculate')



if __name__=="__main__":
    main()