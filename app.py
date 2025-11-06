import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tavily import TavilyClient
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
    }

    .stApp {
        background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: white !important;
    }
    /* --- Hide the default Streamlit top header --- */
    header[data-testid="stHeader"] {
    display: none !important;
    }
            



    .gradient-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .gradient-title {
    color: white !important;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-align: center;
    }


    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea !important;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: rgba(255, 255, 255, 0.6) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    .alert-normal {
        background: rgba(34, 197, 94, 0.2);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .alert-warning {
        background: rgba(249, 115, 22, 0.2);
        border-left: 4px solid #f97316;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .alert-critical {
        background: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #2d3e50 100%);
    }

    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    }

    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    code {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #a8daff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="gradient-header">
        <h1 class="gradient-title" style="color: white;">⚙️ AI-Driven Predictive Maintenance System Using Time-Series Sensor Data</h1>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 🚀 Navigation")
    page = st.radio(
        "",
        ["1️⃣ Overview", "2️⃣ Model Results", "3️⃣ Maintenance Alerts", "4️⃣ Chatbot"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Model Type", "BiLSTM")
    st.metric("Test R²", "0.9438")
    st.metric("Features", "20")

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------- Build BiLSTM Model ----------------------
def build_bilstm_model(timesteps, features):
    optimizer = Adam(learning_rate=0.0005)

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, features)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # Output for RUL
    ])

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae']
    )
    return model

if page == "1️⃣ Overview":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📖 Project Overview")
    st.markdown("""
        This system predicts **Remaining Useful Life (RUL)** of engines using time-series sensor data
        and alerts maintenance teams proactively.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- File Upload ----------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Sensor Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV sensor data",
        type=['csv'],
        help="Upload a CSV file containing sensor readings"
    )

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- Features ----------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Input Features")
    features = [
        'op_setting_1', 'op_setting_2', 'op_setting_3',
        's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9',
        's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17',
        's_20', 's_21'
    ]
    cols = st.columns(4)
    for idx, feature in enumerate(features):
        with cols[idx % 4]:
            st.markdown(f"- `{feature}`")
    st.markdown(f"### 📊 Total Features: **{len(features)}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- Predict Section ----------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Predict Remaining Useful Life (RUL)")

    if df is not None:
        if st.button("🔮 Predict RUL"):
            try:
                SEQ_LENGTH = 50  # model expects 50 timesteps

                if len(df) < SEQ_LENGTH:
                    st.warning(f"⚠ Please upload at least {SEQ_LENGTH} rows for prediction.")
                else:
                    # Take the last 50 rows for prediction
                    X_input = df[features].iloc[-SEQ_LENGTH:].values
                    X_input = X_input.reshape(1, SEQ_LENGTH, len(features))  # (1, timesteps, features)

                    # Build model and load weights
                    model = build_bilstm_model(SEQ_LENGTH, len(features))
                    model.load_weights(r"C:\MILESTONE_5\final_rul_bilstm.weights.h5")

                    # Predict
                    y_pred = model.predict(X_input)
                    predicted_rul = float(y_pred[0][0])

                    # Alert thresholds
                    CRITICAL_THRESHOLD = 20
                    WARNING_THRESHOLD = 40

                    def get_alert_level(rul_value):
                        if rul_value < CRITICAL_THRESHOLD:
                            return "CRITICAL", "🚨"
                        elif rul_value < WARNING_THRESHOLD:
                            return "WARNING", "⚠"
                        else:
                            return "NORMAL", "✅"

                    alert_level, icon = get_alert_level(predicted_rul)

                    # Display
                    st.subheader("🔧 Prediction Result")
                    st.metric(
                        label="Predicted Remaining Useful Life (RUL)",
                        value=f"{predicted_rul:.2f} cycles"
                    )
                    st.success(f"{icon} Status: **{alert_level}**")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("⚠ Please upload a CSV file to make predictions.")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "2️⃣ Model Results":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("Test RMSE", "7.9447", col1),
        ("Test MAE", "5.3960", col2),
        ("Test R²", "0.9438", col3),
        ("Mean Bias", "0.0178", col4),
        ("Error Std", "7.9447", col5)
    ]

    for label, value, col in metrics:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
        The **BiLSTM (Bidirectional Long Short-Term Memory)** model captures both past and future
        temporal dependencies in sensor data, enabling accurate RUL prediction.

        **Key Advantages:**
        - **Bidirectional Processing**: Learns from both past and future sequences
        - **Temporal Dependencies**: Captures long-term patterns in sensor data
        - **High Accuracy**: Achieves 94.38% R² score on test data
        - **Robust Predictions**: Low mean bias (0.0178) ensures unbiased estimates
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance Visualization")

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R²', 'Mean Bias', 'Error Std'],
        'Value': [7.9447, 5.3960, 0.9438, 0.0178, 7.9447],
        'Type': ['Error', 'Error', 'Score', 'Bias', 'Error']
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Value'],
        marker=dict(
            color=['#ef4444', '#ef4444', '#22c55e', '#3b82f6', '#ef4444'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
        ),
        text=metrics_df['Value'],
        textposition='outside',
        textfont=dict(color='white')
    ))

    fig.update_layout(
    title="Model Performance Metrics",
    title_font=dict(size=20, color='white'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        title=dict(text="Metric", font=dict(color='white')),
        tickfont=dict(color='white'),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        title=dict(text="Value", font=dict(color='white')),
        tickfont=dict(color='white'),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    height=400
)


    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Detailed Metrics Table")

    detailed_metrics = pd.DataFrame({
        'Metric': ['Test RMSE', 'Test MAE', 'Test R²', 'Mean Bias', 'Error Std'],
        'Value': [7.9447, 5.3960, 0.9438, 0.0178, 7.9447],
        'Description': [
            'Root Mean Square Error - Average prediction error magnitude',
            'Mean Absolute Error - Average absolute deviation',
            'R-squared Score - Proportion of variance explained (94.38%)',
            'Mean Bias - Average prediction bias (near zero is best)',
            'Error Standard Deviation - Prediction uncertainty measure'
        ]
    })

    st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "3️⃣ Maintenance Alerts":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🚨 Alert Threshold Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="alert-critical">
                <h3>🚨 CRITICAL THRESHOLD</h3>
                <h2>20 cycles</h2>
                <p>Immediate maintenance required</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ WARNING THRESHOLD</h3>
                <h2>40 cycles</h2>
                <p>Schedule maintenance soon</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Alert Statistics Across Test Set")

    st.code("""--- Alert Statistics Across Test Set ---
Total test samples: 3156
  ✓ NORMAL:    2376 (75.3%)
  ⚠ WARNING:    391 (12.4%)
  🚨 CRITICAL:   389 (12.3%)""", language=None)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Alert Distribution")

    alert_data = pd.DataFrame({
        'Status': ['NORMAL', 'WARNING', 'CRITICAL'],
        'Count': [2376, 391, 389],
        'Percentage': [75.3, 12.4, 12.3]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=alert_data['Status'],
        y=alert_data['Count'],
        marker=dict(
            color=['#22c55e', '#f97316', '#ef4444'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
        ),
        text=[f"{count}<br>({pct}%)" for count, pct in zip(alert_data['Count'], alert_data['Percentage'])],
        textposition='outside',
        textfont=dict(color='white', size=14)
    ))

    fig.update_layout(
    title="Alert Status Distribution (3156 Total Samples)",
    title_font=dict(size=20, color='white'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        title=dict(text="Alert Status", font=dict(color='white')),
        tickfont=dict(color='white', size=14),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        title=dict(text="Number of Samples", font=dict(color='white')),
        tickfont=dict(color='white'),
        gridcolor='rgba(255,255,255,0.1)'
    ),
    height=450
)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🥧 Alert Proportion")

    fig_pie = go.Figure(data=[go.Pie(
        labels=alert_data['Status'],
        values=alert_data['Count'],
        hole=0.4,
        marker=dict(colors=['#22c55e', '#f97316', '#ef4444']),
        textfont=dict(color='white', size=14),
        textinfo='label+percent'
    )])

    fig_pie.update_layout(
        title="Alert Distribution Breakdown",
        title_font=dict(size=20, color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(255,255,255,0.1)'
        ),
        height=400
    )

    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Alert Guidelines")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="alert-normal">
                <h4>✓ NORMAL</h4>
                <p><strong>RUL > 40 cycles</strong></p>
                <p>Continue regular monitoring</p>
                <p>No immediate action needed</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="alert-warning">
                <h4>⚠️ WARNING</h4>
                <p><strong>20 < RUL ≤ 40 cycles</strong></p>
                <p>Schedule maintenance within next cycle</p>
                <p>Increased monitoring recommended</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="alert-critical">
                <h4>🚨 CRITICAL</h4>
                <p><strong>RUL ≤ 20 cycles</strong></p>
                <p>Immediate maintenance required</p>
                <p>High failure risk</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "4️⃣ Chatbot":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🤖 AI Assistant - Ask Me Anything!")
    st.markdown("""
        Ask questions about RUL, BiLSTM models, predictive maintenance, or how the alert system works.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm your AI assistant for the Predictive Maintenance System. Ask me anything about RUL prediction, BiLSTM models, maintenance alerts, or sensor data analysis!"
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    tavily_api_key = st.secrets["TAVILY_API_KEY"]
                    tavily_client = TavilyClient(api_key=tavily_api_key)

                    # Short query to avoid 400 char limit
                    query = f"{prompt} predictive maintenance RUL BiLSTM"
                    query = query[:350]

                    response = tavily_client.search(query=query, max_results=1)

                    # ✅ Start crafting proper answer
                    answer = ""

                    if "rul" in prompt.lower():
                        answer += (
                            "**Remaining Useful Life (RUL)** tells how many cycles an engine can run "
                            "before failure. It helps schedule maintenance early to avoid breakdowns.\n\n"
                        )

                    if "machine" in prompt.lower() or "ml" in prompt.lower():
                        answer += (
                            "**Machine Learning (ML)** helps computers learn from past data instead of "
                            "being manually programmed with rules. In RUL prediction, models learn "
                            "patterns from sensor data to forecast engine health.\n\n"
                        )

                    if "bilstm" in prompt.lower() or "lstm" in prompt.lower():
                        answer += (
                            "**BiLSTM (Bidirectional LSTM)** reads engine sensor sequences both forward "
                            "and backward. This helps it understand degradation patterns better, "
                            "improving prediction accuracy.\n\n"
                        )

                    if "alert" in prompt.lower():
                        answer += (
                            "**Maintenance Alerts**\n"
                            "- 🚨 Critical: RUL ≤ 20 cycles → Immediate service\n"
                            "- ⚠️ Warning: RUL ≤ 40 cycles → Schedule soon\n"
                            "- ✅ Normal: RUL > 40 cycles → No action required\n\n"
                        )

                    if "sensor" in prompt.lower() or "feature" in prompt.lower():
                        answer += (
                            "Our model uses **20 sensor features**, including temperature, pressure, "
                            "and vibration readings to evaluate engine health.\n\n"
                        )

                    # ✅ If nothing matched above, give a general fallback
                    if not answer:
                        answer = (
                            "I can help you with questions about RUL, predictive maintenance, BiLSTM "
                            "models, alerts, or sensor features. Try asking about one of those topics! 😊"
                        )

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    fallback_msg = (
                        "I can help with questions on:\n\n"
                        "✅ What is RUL?\n"
                        "✅ How does BiLSTM work?\n"
                        "✅ How alerts are triggered?\n"
                        "✅ Which sensors are used?\n\n"
                        "Ask me anything! 😊"
                    )
                    st.markdown(fallback_msg)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                    st.warning("Using fallback mode (Tavily error)")
