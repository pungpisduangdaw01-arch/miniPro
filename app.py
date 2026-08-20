import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ระบบวิเคราะห์การออกกลางคันของนักศึกษา", layout="wide")

import streamlit as st
import os

# เช็กตำแหน่งไฟล์รูปในโฟลเดอร์ img
image_path = "img/aa.jpg"

if os.path.exists(image_path):
    st.sidebar.image(image_path, width=150)
else:
    # หากหาไฟล์ไม่เจอ ให้แสดงรูปภาพสำรองชั่วคราว
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150) # เปลี่ยนเป็น URL รูปของคุณ หรือใช้ st.image("profile.jpg")
st.sidebar.title("👨‍💻 ข้อมูลผู้พัฒนา")
st.sidebar.write("**ชื่อ-นามสกุล:/nนาย ชิษณุพงศ์ เกตุพูนทอง") 
st.sidebar.write("**รหัสนักศึกษา:** 664245004")          
st.sidebar.write("**หมู่เรียน:** 66/43")                 
st.sidebar.markdown("---")

st.title("🎓 ระบบวิเคราะห์ความเสี่ยงการออกกลางคัน (Student Dropout Prediction)")
st.write("โปรแกรมวิเคราะห์ปัจจัยเสี่ยงในการออกกลางคันของนักศึกษาด้วย Machine Learning")

@st.cache_resource
def load_model():
    return joblib.load('dropout_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ข้อมูลทั่วไป")
    age = st.number_input("อายุ", min_value=15, max_value=60, value=20)
    gender = st.selectbox("เพศ", ["Male", "Female"])
    department = st.selectbox("คณะ/สาขา", ["Engineering", "Arts", "Business", "Science", "Other"])
    semester = st.selectbox("ชั้นปี", ["Year 1", "Year 2", "Year 3", "Year 4"])
    parental_edu = st.selectbox("การศึกษาผู้ปกครอง", ["High School", "Bachelor", "Master", "PhD"])

with col2:
    st.subheader("ผลการเรียน & พฤติกรรม")
    gpa = st.number_input("GPA สะสม", 0.0, 4.0, 2.5)
    semester_gpa = st.number_input("GPA เทอมล่าสุด", 0.0, 4.0, 2.5)
    cgpa = st.number_input("CGPA", 0.0, 4.0, 2.5)
    attendance = st.slider("อัตราการเข้าเรียน (%)", 0.0, 100.0, 80.0)
    study_hours = st.number_input("เวลาอ่านหนังสือ/วัน (ชม.)", 0.0, 24.0, 3.0)
    delay_days = st.number_input("ส่งงานช้าเฉลี่ย (วัน)", 0, 30, 1)

with col3:
    st.subheader("สภาพแวดล้อม & สุขภาพ")
    income = st.number_input("รายได้ครอบครัว", 0.0, 500000.0, 30000.0)
    travel_time = st.number_input("เวลาเดินทาง (นาที)", 0.0, 300.0, 30.0)
    stress = st.slider("ระดับความเครียด (0-10)", 0.0, 10.0, 5.0)
    internet = st.selectbox("การเข้าถึงอินเทอร์เน็ต", ["Yes", "No"])
    part_time = st.selectbox("ทำงานพาร์ทไทม์", ["No", "Yes"])
    scholarship = st.selectbox("ได้รับทุนการศึกษา", ["No", "Yes"])

if st.button("🔍 วิเคราะห์ความเสี่ยง"):
    input_data = pd.DataFrame([{
        'Age': age, 'Gender': gender, 'Family_Income': income,
        'Internet_Access': internet, 'Study_Hours_per_Day': study_hours,
        'Attendance_Rate': attendance, 'Assignment_Delay_Days': delay_days,
        'Travel_Time_Minutes': travel_time, 'Part_Time_Job': part_time,
        'Scholarship': scholarship, 'Stress_Index': stress, 'GPA': gpa,
        'Semester_GPA': semester_gpa, 'CGPA': cgpa, 'Semester': semester,
        'Department': department, 'Parental_Education': parental_edu
    }])
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ **มีความเสี่ยงสูงที่จะออกกลางคัน** (ความน่าจะเป็น: {prob:.1f}%)")
    else:
        st.success(f"✅ **มีความเสี่ยงต่ำที่จะออกกลางคัน** (โอกาสเสี่ยง: {prob:.1f}%)")