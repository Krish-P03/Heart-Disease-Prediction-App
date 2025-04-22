import streamlit as st
import pandas as pd
import pickle
import os
import datetime
import numpy as np
import altair as alt
import subprocess

MODEL_PATH = 'models/model.pkl'
USER_DATA_PATH = 'models/user_inputs.csv'
RETRAIN_TRIGGER = 10


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def save_user_input(input_dict):
    df = pd.DataFrame([input_dict])
    if os.path.exists(USER_DATA_PATH):
        df_existing = pd.read_csv(USER_DATA_PATH)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(USER_DATA_PATH, index=False)


def retrain_if_needed():
    if not os.path.exists(USER_DATA_PATH):
        return
    df = pd.read_csv(USER_DATA_PATH)
    if len(df) >= RETRAIN_TRIGGER:
        orig = pd.read_csv('data/heart.csv')
        merged = pd.concat([orig, df], ignore_index=True)
        merged.to_csv('data/heart.csv', index=False)
        subprocess.run(['python', 'preprocess_and_train.py'])


def get_table_download_link(df, file_label='Download CSV'):
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label=file_label,
        data=csv,
        file_name='health_history.csv',
        mime='text/csv'
    )


def get_pdf_download_link(df):
    import pdfkit
    try:
        html = df.to_html(index=False)
        pdf = pdfkit.from_string(html, False)
        st.download_button(
            label="Download PDF Report",
            data=pdf,
            file_name='health_report.pdf',
            mime='application/pdf'
        )
    except ImportError:
        st.warning(
            'PDF download requires pdfkit and wkhtmltopdf. '
            'Please install them for PDF export.'
        )
    except Exception as e:
        st.error(f'PDF export failed: {e}')


def reset_history():
    if os.path.exists(USER_DATA_PATH):
        os.remove(USER_DATA_PATH)
        st.success('History cleared!')


def main():
    st.set_page_config(
        page_title="❤️ Heart Health Tracker",
        page_icon="❤️",
        layout="wide"
    )

    st.markdown("""
    <h2 style='color:#d7263d;'>🫀 Heart Disease Prediction App</h2>
    <p style='font-size:18px;'>
        Enter your health data below. Get instant risk percentage and track
        your progress over time!
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader('Enter Patient Data')
        fields = {
            'age': st.number_input(
                'Age',
                min_value=1,
                max_value=120,
                value=50,
                help='Age in years'
            ),
            'sex': st.selectbox(
                'Sex',
                options=[1, 0],
                format_func=lambda x: 'Male' if x == 1 else 'Female'
            ),
            'cp': st.selectbox(
                'Chest Pain Type (cp)',
                options=[0, 1, 2, 3]
            ),
            'trestbps': st.number_input(
                'Resting Blood Pressure (trestbps)',
                min_value=80,
                max_value=250,
                value=120
            ),
            'chol': st.number_input(
                'Cholesterol (chol)',
                min_value=100,
                max_value=600,
                value=200
            ),
            'fbs': st.selectbox(
                'Fasting Blood Sugar > 120 mg/dl (fbs)',
                options=[1, 0]
            ),
            'restecg': st.selectbox(
                'Resting ECG (restecg)',
                options=[0, 1, 2]
            ),
            'thalach': st.number_input(
                'Max Heart Rate (thalach)',
                min_value=60,
                max_value=250,
                value=150
            ),
            'exang': st.selectbox(
                'Exercise Induced Angina (exang)',
                options=[1, 0]
            ),
            'oldpeak': st.number_input(
                'ST depression (oldpeak)',
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            ),
            'slope': st.selectbox(
                'Slope', options=[0, 1, 2]
            ),
            'ca': st.selectbox(
                'Number of Major Vessels (ca)',
                options=[0, 1, 2, 3]
            ),
            'thal': st.selectbox(
                'Thalassemia (thal)',
                options=[1, 2, 3]
            ),
        }
        today = datetime.date.today().isoformat()
        fields['date'] = today

    with col2:
        st.subheader('Risk Result')
        if st.button('💡 Predict'):
            try:
                model = load_model()
                input_df = pd.DataFrame([fields])
                drop_cols = [
                    col for col in input_df.columns
                    if col not in [
                        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca', 'thal', 'date'
                    ]
                ]
                input_df = input_df.drop(columns=drop_cols, errors='ignore')
                if hasattr(model, 'predict_proba'):
                    input_features = input_df.drop('date', axis=1)
                    probabilities = model.predict_proba(input_features)
                    risk_percent = float(probabilities[0][1]) * 100
                else:
                    input_features = input_df.drop('date', axis=1)
                    prediction = model.predict(input_features)[0]
                    risk_percent = 100.0 if prediction == 1 else 0.0
                if risk_percent < 50:
                    color = '#1bc47d'
                    emoji = '😃'
                    msg = 'Low Risk'
                else:
                    color = '#d7263d'
                    emoji = '⚠️'
                    msg = 'High Risk'
                st.markdown(f"""
                <div style='padding:20px; border-radius:10px; '
                     'background:#f9f9f9; '
                     f'border:2px solid {color};'>
                    <h3 style='color: {color};'>{emoji} {msg}</h3>
                    <p style='font-size:22px;'>
                        Your heart disease risk: <b>{risk_percent:.1f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                retrain_data = dict(fields)
                retrain_data['target'] = 1 if risk_percent >= 50 else 0
                save_user_input(retrain_data)
                retrain_if_needed()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown('---')
    st.subheader('📊 Your Prediction History')
    if os.path.exists(USER_DATA_PATH):
        hist_df = pd.read_csv(USER_DATA_PATH)
        if 'risk_percent' not in hist_df.columns:
            hist_df['risk_percent'] = hist_df['target'].map(
                lambda x: 100.0 if x == 1 else 0.0
            )
        st.markdown(
            "<h4 style='color:#1bc47d;'>Health Summary</h4>",
            unsafe_allow_html=True
        )
        total = len(hist_df)
        risk = int(hist_df['target'].sum())
        safe = total - risk
        st.write(
            f"Total Records: {total} | 🚨 At Risk: {risk} | ✅ No Risk: {safe}"
        )
        unique_risks = hist_df['risk_percent'].unique()
        if (len(unique_risks) > 2 or
                (0 not in unique_risks or 100 not in unique_risks)):
            st.markdown('**Risk Percentage Distribution**')
            hist_chart = alt.Chart(hist_df).mark_bar().encode(
                alt.X(
                    'risk_percent',
                    bin=alt.Bin(maxbins=20),
                    title='Predicted Risk (%)'
                ),
                y='count()',
                tooltip=['count()']
            ).properties(width=500, height=250)
            st.altair_chart(hist_chart)

        def risk_level(r):
            return 'Low' if r < 50 else 'High'
        hist_df['risk_level'] = hist_df['risk_percent'].apply(risk_level)
        pie_data = hist_df['risk_level'].value_counts().reset_index()
        pie_data.columns = ['Risk Level', 'Count']
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=40).encode(
            theta=alt.Theta(field='Count', type='quantitative'),
            color=alt.Color(field='Risk Level', type='nominal'),
            tooltip=['Risk Level', 'Count']
        )
        st.altair_chart(pie_chart, use_container_width=True)
        st.dataframe(hist_df.tail(20))
        get_table_download_link(
            hist_df,
            file_label='Download Full History (CSV)'
        )
    if os.path.exists(USER_DATA_PATH):
        if st.button('🗑️ Clear History'):
            reset_history()
    else:
        st.info('No history yet. Predictions will appear here!')


if __name__ == '__main__':
    main()
