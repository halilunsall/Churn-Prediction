import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
import os

# Creating PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# To add date to the title of PDF file
from datetime import datetime

# Creating Word
from docx import Document
from docx.shared import Inches

def generate_pdf(text,filename='output.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    paragraphs = [Paragraph("<font size='12'>" + p.strip() + "</font>", style) for p in text.split("\n\n")]
    flowables = []
    for p in paragraphs:
        flowables.append(p)
        flowables.append(Spacer(1, 12))
    doc.build(flowables)
    return doc

def generate_filename():
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    filename = f"employee_churn_{hour}_{minute}.pdf"
    return filename

def html_options(text=None, align="left", size=12, weight="normal", style="normal", color="#F4A460", bg_color=None, bg_size=16, on='main', to_link=None, image_width=None, image_height=None, image_source=None, image_bg_color=None):
    if on == 'main':
        st.markdown(f"""<div style="background-color:{bg_color};padding:{bg_size}px">
        <h2 style='text-align: {align}; font-size: {size}px; font-weight: {weight}; font-style: {style}; color: {color};'>{text} </h2>
        </div>""", unsafe_allow_html=True)
    elif on == 'side':
        st.sidebar.markdown(f"""<div style="background-color:{bg_color};padding:{bg_size}px">
        <h2 style='text-align: {align}; font-size: {size}px; font-weight: {weight}; font-style: {style}; color: {color};'>{text} </h2>
        </div>""", unsafe_allow_html=True)
    elif on == 'image':
        image_style = f"background-color:{image_bg_color};" if image_bg_color else ""
        st.markdown(f"""<div style="text-align: {align};"><img width="{image_width}" height="{image_height}" src="{image_source}" style="{image_style}" /></a></div>""", unsafe_allow_html=True)
    elif on == 'link':
        image_style = f"background-color:{image_bg_color};" if image_bg_color else ""
        st.markdown(f"""<div style="text-align: {align};"> <a href="{to_link}"><img width="{image_width}" height="{image_height}" src="{image_source}" style="{image_style}" /></a></div>""", unsafe_allow_html=True)

html_options(text='Please do not forget to change the theme you have chosen with the settings option in the upper right corner.', on='side', align='center', size=15, weight='bold')
theme = st.sidebar.selectbox('', ['Dark', 'Light'])

statistical_findings = [
  "There is a significant difference in average values between employees who left and those who stayed for column satisfaction_level.",
  "There is no significant difference in average values between employees who left and those who stayed for column last_evaluation.",
  "There is no significant difference in average values between employees who left and those who stayed for column number_project.",
  "There is a significant difference in average values between employees who left and those who stayed for column average_montly_hours.",
  "There is a significant difference in average values between employees who left and those who stayed for column time_spend_company.",
  "There is a significant difference in average values between employees who left and those who stayed for column Work_accident.",
  "There is a significant difference in average values between employees who left and those who stayed for column promotion_last_5years.",
  "There is a significant difference in average values between employees who left and those who stayed for column left.",
  "There is evidence to suggest a significant difference in the proportion of employees who left the company based on whether they had a work accident or not.",
  "There is a significant difference in the average satisfaction level between employees who had a work accident and those who didn't.",
  "There is a statistically significant association between the salary level of employees and the likelihood of them leaving the company."
]

# HEAD TO PICTURE
if theme=='Light':
    st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_light/Employee%20churn%20prediction.png")
else:
    st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_dark/Employee%20churn%20prediction.png")
st.write('')
st.write('')


# SIDEBAR 
with st.sidebar:
    if theme=='Light':
        st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_light/Employee%20Information.png")
    else:
        st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_dark/Employee%20Information.png")
    st.write('')
    Departments = st.selectbox('Department', ['Select','Sales', 'Technical', 'Support', 'IT', 'Research and Development','Product Manager', 'Marketing', "Accounting", "Human Resources", "Management", "Others"])
    st.write('')
    salary = st.selectbox('Salary Status', ['Select','Low', 'Medium', 'High'])
    st.write('')
    satisfaction_level = st.slider('Satisfaction Level', min_value=0.0, max_value=1.0, value=0.09, step=0.01)
    st.write('')
    last_evaluation = st.slider('Last Evaluation(from Employer)', min_value=0.0, max_value=1.0, value=0.79, step=0.01)
    st.write('')
    number_project = st.slider('Assigned Project', min_value=0, max_value=10, value=6, step=1)
    st.write('')
    average_montly_hours = st.slider('Monthly Working Time(Hour)', min_value=50, max_value=400, value=293, step=1)
    st.write('')
    time_spend_company = st.slider('Time in the Company(Year)', min_value=1, max_value=10, value=5, step=1)
    st.write('')
    Work_accident = st.radio('Work Accident', ('True', 'False'))
    st.write('')
    promotion_last_5years = st.radio('Get Promoted(Last 5 Year)', ('True', 'False'))


# Create a dataframe using feature inputs
show_df = {'Informations':{'Departments': Departments,
        'Salary': salary,
        'Satisfaction Level': satisfaction_level,
        'Last Evaluation': last_evaluation,
        'Assigned Project': number_project,
        'Monthly Working Time': average_montly_hours,
        'Time in the Company': time_spend_company,
        'Work Accident': Work_accident,
        'Get Promoted': promotion_last_5years}}


# LOAD MODEL

model_df = pd.DataFrame(data=[[satisfaction_level, last_evaluation, number_project, 
                                average_montly_hours, time_spend_company, Work_accident, 
                                promotion_last_5years, Departments, salary]],
                        columns=['satisfaction_level', 'last_evaluation', 'number_project', 
                                'average_montly_hours', 'time_spend_company', 'Work_accident', 
                                'promotion_last_5years', 'departments', 'salary'])
depts_map = {"Sales":"sales" , "Technical":"technical" , "Support":"support" , "IT":"IT" , 
       "Research and Development":"RandD" , "Product Manager":"product_mng" , "Marketing":"marketing" , 
        "Accounting":"accounting" , "Human Resources":"hr" , "Management":"management", "Select":"Select", "Others":"Others"}
salary_map = {"Low":1 , "Medium":2 , "High":0, "Select":"Select", "Other":"Others"}
bool_map = {"False":0, 'True':1}

model_df['promotion_last_5years'] = model_df['promotion_last_5years'].map(bool_map)
model_df['Work_accident'] = model_df['Work_accident'].map(bool_map)
model_df['departments'] = model_df['departments'].map(depts_map)
model_df['salary'] = model_df['salary'].map(salary_map)
model_churn = pickle.load(open('emp_churn_final_model', 'rb'))



# Employee TABLE
html_options(text='Employee', size=40, weight='bold', color='#FFA500', align='center')
st.write('')
st.table(show_df)

# ChatGPT
with open("openai_api.txt") as file:
    openai.api_key = file.read()


messages = [{"role":"system", "content":"Write like a world-known expert HR manager. Do not mention that you are HR manager."}]

def AdviceGPT():
    with st.spinner('⚡Gearing up to blow your mind...'):
        leave_text = ''
        if result == 1:
            leave_text = f'This employee is churn according to ml model with {result_proba[1]} score'
        else:
            leave_text = f'This employee is not churn according to ml model with {result_proba[0]} score'
        show_df['Informations']['Monthly Working Time'] = str(show_df['Informations']['Monthly Working Time'])  + ' hours'
        message = f"How can I increase the productivity of this employee? Employee information: {show_df}. {leave_text}. These are statistical test results based on hypothesis tests:{' '.join(statistical_findings)}. Consider employee information and evaluate each information. Include numbers and values. Also comment on churn with the ML score rounded. Write engaging conclusion."
        if message:
            messages.append({"role":"system", "content":message})
            response = openai.ChatCompletion.create(
                        model = "gpt-3.5-turbo",
                        messages = messages
        )
        gpt_reply = response["choices"][0]["message"]["content"]
        messages.append({"role":"system", "content":gpt_reply})
    st.success(gpt_reply)

    filename = generate_filename()
    generate_pdf(gpt_reply,filename)
    with open(filename, "rb") as f:
        st.download_button("Download PDF", f.read(), file_name=filename, mime="application/pdf")


messages_cust = [{"role":"system", "content":"Write as if you are the customer support team of a company and imagine that you are talking to the customer. And your name is Tony."}]

def CustomGPT():

    message = st.text_input("")

    if message:
        with open("messages.txt", "a") as file:
            file.write(message+"\n")
        messages_cust.append({"role":"system", "content":message})
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages_cust
            )
        gpt_reply = response["choices"][0]["message"]["content"]

        with open("replys.txt", "a") as file:
            file.write(gpt_reply+ "\n")

        messages_cust.append({"role":"system", "content":gpt_reply})


# PREDICTION
result = np.nan
result_proba = np.nan
predict = st.button("Predict",type='primary', use_container_width=True)
if predict:
    if Departments == "Select" or salary == 'Select':
        st.warning('Be sure to enter department and salary information.')
    else:
        result = model_churn.predict(model_df)[0]
        result_proba = model_churn.predict_proba(model_df)[0]
        if result == 1:
            if theme=='Light':
                st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_light/true.png")
            else:
                st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_dark/true.png")
            AdviceGPT()
        else:
            if theme=='Light':
                st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_light/false.png")
            else:
                st.image("https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_dark/false.png")
            AdviceGPT()


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
if theme=='Light':
    html_options(on='image', image_source='https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_light/Contact%20LIVEL.png', align='center', image_width=200)
else:
    html_options(on='image', image_source='https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/images_dark/Contact%20LIVED.png', align='center', image_width=200)
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
chat = col5.button(':green[CHAT]')


CustomGPT()
with open("messages.txt", "r") as file:
    mes = file.read()
    st.info("User: "+mes.strip().split('\n')[-1])
with open("replys.txt", "r") as file:
    rep = file.read()
    st.info("Customer Support: "+rep.strip().split('\n')[-1])

st.write('')

col1, col2, col3 = st.columns(3)

if theme=='Light':
    with col1:
        html_options(on='link', to_link='https://www.linkedin.com/in/halilibrahimunsal/', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/linkedin1.png")
    with col2:
        html_options(on='link', to_link='https://github.com/halilunsall', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/git1.png")
    with col3:
        html_options(on='link', to_link='https://public.tableau.com/app/profile/halilunsal', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/tableau1.png")
else:
    with col1:
        html_options(on='link', to_link='https://www.linkedin.com/in/halilibrahimunsal/', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/linkedin.png")
    with col2:
        html_options(on='link', to_link='https://github.com/halilunsall', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/git.png")
    with col3:
        html_options(on='link', to_link='https://public.tableau.com/app/profile/halilunsal', image_height=60, image_width=60, image_source="https://raw.githubusercontent.com/halilunsall/Churn-Prediction/main/logos/tableau.png")