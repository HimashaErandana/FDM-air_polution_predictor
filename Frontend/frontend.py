import streamlit as st
import requests

st.markdown("""
    <style>
    .main {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 15px;
    }
    .stForm {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .stNumberInput, .stSelectbox {
        border-radius: 12px ;
    }
    .success-box {
        background-color: #dcfce7;
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid #16a34a;
    }
    .error-box {
        background-color: #fee2e2;
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid #dc2626;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Air Pollution Predictor")

with st.form("user_input_form"):
    
    col1, col2 = st.columns(2)
    with col1:
        vehicle_ownership = st.selectbox("Vehicle Ownership", ['Two-wheeler', 'Public Only', 'Car'])
        location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])
        green_space_access = st.selectbox("Green Space Access", ['Moderate', 'High', 'Low'])
        work_location_type = st.selectbox("Work Location Type", ['Factory', 'Fieldwork', 'Remote', 'Office'])
        smoker_in_household = st.selectbox("Smoker in Household", ["Yes", "No"])
        use_of_air_purifiers = st.selectbox("Use of Air Purifiers", ["Yes", "No"])
    with col2:
        awareness_level = st.selectbox("Awareness Level", ['Medium', 'Low', 'High'])
        daily_travel_time = st.number_input("Daily Travel Time (minutes)", min_value=0, max_value=300, step=1)
        nearby_industries = st.number_input("Nearby Industries",min_value=0, max_value=10, step=1)
        home_air_quality = st.number_input("Home Air Quality", min_value=0, max_value=100, step=1)
        noise_pollution_level = st.number_input("Noise Pollution Level (dB)", min_value=0, max_value=150, step=1)
        years_in_location = st.number_input("Years in Location", min_value=0, max_value=50, step=1)
        

    submitted = st.form_submit_button("Submit")

    if submitted:
        entry = {
            "daily_travel_time": daily_travel_time,
            "vehicle_ownership": vehicle_ownership,
            "location_type": location_type,
            "nearby_industries": nearby_industries,
            "green_space_access": green_space_access,
            "home_air_quality": home_air_quality,
            "work_location_type": work_location_type,
            "smoker_in_household": smoker_in_household,
            "noise_pollution_level": noise_pollution_level,
            "use_of_air_purifiers": use_of_air_purifiers,
            "awareness_level": awareness_level,
            "years_in_location": years_in_location,
        }

        #st.write(entry)
        
        try:
            response = requests.post(
                'http://127.0.0.1:8000/predict',  
                json=entry
            )

            if response.status_code == 200:
                #st.success("Data sent successfully!")
                #st.json(response.json())
                result=response.json()
                prediction=result.get('prediction')

                if prediction is not None:
                    st.markdown(
                        f"""
                            <div class='success-box'>
                                <h4>✅ Prediction result: </h4>
                                <p><b>Risk Analysis : </b>{prediction}</p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No Prediction Value Find")
            else:
                st.error(f"Failed to send data: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
   


