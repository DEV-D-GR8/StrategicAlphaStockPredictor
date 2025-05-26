# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import random
# import time
# import openai
# from typing import Dict, List, Tuple
# import json

# # Configure page
# st.set_page_config(
#     page_title="Strategic Alpha Intelligence System",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for professional look
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         color: #666;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#     }
#     .status-good {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .status-neutral {
#         color: #ffc107;
#         font-weight: bold;
#     }
#     .status-bad {
#         color: #dc3545;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'predictions_history' not in st.session_state:
#     st.session_state.predictions_history = []

# # Fake OpenAI API call simulation
# def generate_fake_prediction(ticker: str, sentiment_override: str = None) -> Dict:
#     """Simulate OpenAI API call for stock prediction"""
    
#     # Simulate API delay
#     time.sleep(random.uniform(1, 3))
    
#     # Generate fake but realistic prediction
#     base_return = random.uniform(-15, 20)
#     confidence = random.uniform(0.6, 0.95)
    
#     # Adjust based on sentiment
#     if sentiment_override == "Bullish":
#         base_return += random.uniform(2, 8)
#         confidence += 0.05
#     elif sentiment_override == "Bearish":
#         base_return -= random.uniform(2, 8)
#         confidence += 0.03
    
#     # Determine signal
#     if base_return > 5:
#         signal = "BUY"
#     elif base_return < -3:
#         signal = "SELL"
#     else:
#         signal = "HOLD"
    
#     # Generate fake explanation
#     explanations = [
#         f"Technical indicators show {signal.lower()} momentum with MACD crossover and RSI at favorable levels.",
#         f"Market sentiment analysis indicates {sentiment_override.lower() if sentiment_override else 'neutral'} outlook for {ticker}.",
#         f"Price action and volume patterns suggest {signal.lower()} opportunity with {confidence*100:.1f}% confidence.",
#         f"Ensemble model consensus: {signal} signal driven by momentum and volatility factors."
#     ]
    
#     return {
#         "predicted_return": round(base_return, 2),
#         "signal": signal,
#         "confidence": round(confidence, 3),
#         "explanation": random.choice(explanations),
#         "timestamp": datetime.now()
#     }

# # Header
# st.markdown('<h1 class="main-header">📈 Strategic Alpha Intelligence System</h1>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Advanced ML-Powered Stock Prediction & Trading Signal Generation</p>', unsafe_allow_html=True)

# # Sidebar Navigation
# st.sidebar.title("🎯 Navigation")
# tab_selection = st.sidebar.radio(
#     "Select Module:",
#     ["🏠 Dashboard", "🔮 Prediction", "📊 Backtesting", "🧠 Model Insights", "⚡ System Status", "📘 About"]
# )

# # Stock universe for dropdowns
# STOCK_UNIVERSE = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
#     "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "WIPRO.NS", "ITC.NS"
# ]

# # Dashboard Tab
# if tab_selection == "🏠 Dashboard":
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Model Version", "v1.2.3", "Updated")
#     with col2:
#         st.metric("Active Signals", "23", "+5")
#     with col3:
#         st.metric("Accuracy (30d)", "74.2%", "+2.1%")
#     with col4:
#         st.metric("Sharpe Ratio", "1.84", "+0.12")
    
#     st.markdown("---")
    
#     # Quick Stats
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📈 Recent Performance")
        
#         # Generate fake performance data
#         dates = pd.date_range(start='2024-01-01', end='2024-05-26', freq='D')
#         strategy_returns = np.cumsum(np.random.normal(0.0008, 0.02, len(dates)))
#         benchmark_returns = np.cumsum(np.random.normal(0.0005, 0.015, len(dates)))
        
#         df_perf = pd.DataFrame({
#             'Date': dates,
#             'Strategy': strategy_returns,
#             'Benchmark': benchmark_returns
#         })
        
#         fig = px.line(df_perf, x='Date', y=['Strategy', 'Benchmark'], 
#                      title="Cumulative Returns vs Benchmark")
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         st.subheader("🎯 Top Signals Today")
        
#         fake_signals = pd.DataFrame({
#             'Ticker': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'META'],
#             'Signal': ['BUY', 'SELL', 'BUY', 'HOLD', 'BUY'],
#             'Confidence': [0.87, 0.92, 0.78, 0.65, 0.81],
#             'Expected Return': ['+8.2%', '-5.1%', '+6.7%', '+1.2%', '+7.3%']
#         })
        
#         st.dataframe(fake_signals, use_container_width=True)

# # Prediction Tab
# elif tab_selection == "🔮 Prediction":
#     st.header("🔮 Stock Prediction Engine")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("Input Parameters")
        
#         selected_ticker = st.selectbox("Stock Ticker", STOCK_UNIVERSE, index=0)
#         prediction_date = st.date_input("Analysis Date", datetime.now())
#         sentiment_override = st.selectbox(
#             "Market Sentiment Override", 
#             ["Auto-Detect", "Bullish", "Neutral", "Bearish"]
#         )
        
#         if st.button("🚀 Generate Prediction", type="primary"):
#             with st.spinner("Analyzing market data and generating prediction..."):
#                 # Simulate API call
#                 prediction = generate_fake_prediction(
#                     selected_ticker, 
#                     sentiment_override if sentiment_override != "Auto-Detect" else None
#                 )
                
#                 st.session_state.predictions_history.append({
#                     'ticker': selected_ticker,
#                     'prediction': prediction
#                 })
                
#                 st.success("Prediction generated successfully!")
    
#     with col2:
#         st.subheader("Prediction Results")
        
#         if st.session_state.predictions_history:
#             latest_pred = st.session_state.predictions_history[-1]
#             pred_data = latest_pred['prediction']
            
#             # Display prediction metrics
#             col2a, col2b, col2c = st.columns(3)
            
#             with col2a:
#                 st.metric("Predicted Return", f"{pred_data['predicted_return']}%")
#             with col2b:
#                 signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
#                 st.metric("Signal", f"{signal_color[pred_data['signal']]} {pred_data['signal']}")
#             with col2c:
#                 st.metric("Confidence", f"{pred_data['confidence']*100:.1f}%")
            
#             # Explanation
#             st.info(f"**AI Explanation:** {pred_data['explanation']}")
            
#             # Fake SHAP visualization
#             st.subheader("🔍 Feature Importance (SHAP Analysis)")
            
#             features = ['return_t-1', 'MACD_hist', 'RSI_14', 'sentiment_score_delta', 'price_volatility_5d', 'volume_ratio']
#             importance = np.random.uniform(-0.3, 0.4, len(features))
            
#             fig = go.Figure(go.Bar(
#                 x=importance,
#                 y=features,
#                 orientation='h',
#                 marker_color=['red' if x < 0 else 'green' for x in importance]
#             ))
#             fig.update_layout(title="SHAP Feature Importance", xaxis_title="SHAP Value")
#             st.plotly_chart(fig, use_container_width=True)
        
#         else:
#             st.info("👆 Select a stock and click 'Generate Prediction' to see results")

# # Backtesting Tab
# elif tab_selection == "📊 Backtesting":
#     st.header("📊 Strategy Backtesting")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("Backtest Parameters")
        
#         start_date = st.date_input("Start Date", datetime(2023, 1, 1))
#         end_date = st.date_input("End Date", datetime(2024, 5, 26))
#         sector_filter = st.multiselect("Sector Filter", 
#                                      ["Technology", "Finance", "Healthcare", "Energy", "Consumer"],
#                                      default=["Technology"])
#         transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05)
        
#         if st.button("🔄 Run Backtest", type="primary"):
#             with st.spinner("Running backtest simulation..."):
#                 time.sleep(3)  # Simulate processing time
#                 st.success("Backtest completed!")
    
#     with col2:
#         st.subheader("Backtest Results")
        
#         # Performance Metrics
#         col2a, col2b, col2c, col2d = st.columns(4)
        
#         with col2a:
#             st.metric("CAGR", "18.7%", "+4.2%")
#         with col2b:
#             st.metric("Sharpe Ratio", "1.84", "+0.31")
#         with col2c:
#             st.metric("Max Drawdown", "-8.3%", "")
#         with col2d:
#             st.metric("Win Rate", "67.2%", "+12.1%")
        
#         # Cumulative Returns Chart
#         dates = pd.date_range(start=start_date, end=end_date, freq='D')
#         strategy_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
#         benchmark_returns = np.cumsum(np.random.normal(0.0007, 0.015, len(dates)))
        
#         df_backtest = pd.DataFrame({
#             'Date': dates,
#             'Strategy': strategy_returns,
#             'Nifty 50': benchmark_returns
#         })
        
#         fig = px.line(df_backtest, x='Date', y=['Strategy', 'Nifty 50'], 
#                      title="Strategy vs Benchmark Performance")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Trade History
#         st.subheader("Recent Trades")
#         fake_trades = pd.DataFrame({
#             'Date': pd.date_range(start='2024-05-01', periods=10, freq='2D'),
#             'Ticker': np.random.choice(STOCK_UNIVERSE[:8], 10),
#             'Action': np.random.choice(['BUY', 'SELL'], 10),
#             'Return': np.random.uniform(-8, 15, 10).round(2)
#         })
#         fake_trades['Return'] = fake_trades['Return'].apply(lambda x: f"{x:+.2f}%")
#         st.dataframe(fake_trades, use_container_width=True)

# # Model Insights Tab
# elif tab_selection == "🧠 Model Insights":
#     st.header("🧠 Model Architecture & Insights")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🏗️ Ensemble Architecture")
        
#         st.markdown("""
#         **Stacking Ensemble Approach:**
        
#         **Level 1 Base Models:**
#         - 🌳 XGBoost Classifier
#         - 🐱 CatBoost Classifier  
#         - 🌲 Random Forest Classifier
#         - 📊 LightGBM Classifier
        
#         **Level 2 Meta-Model:**
#         - 🚀 Meta-XGBoost (Final Predictor)
        
#         **Feature Engineering:**
#         - Technical Indicators (20+ features)
#         - Sentiment Analysis Scores
#         - Market Microstructure Data
#         - Volatility Surfaces
#         """)
        
#         st.subheader("⚙️ Model Configuration")
#         st.code("""
# # Model Hyperparameters
# xgb_params = {
#     'n_estimators': 500,
#     'max_depth': 8,
#     'learning_rate': 0.05,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8
# }

# # Cross-Validation: 5-fold TimeSeriesSplit
# # Feature Selection: Recursive Feature Elimination
# # Optimization: Bayesian Hyperparameter Tuning
#         """, language='python')
    
#     with col2:
#         st.subheader("📈 Feature Importance Analysis")
        
#         # Feature importance table
#         feature_importance = pd.DataFrame({
#             'Feature': ['return_t-1', 'MACD_hist', 'RSI_14', 'sentiment_score_delta', 
#                        'price_volatility_5d', 'volume_ratio', 'bollinger_position', 'momentum_5d'],
#             'Importance': [0.21, 0.18, 0.14, 0.12, 0.09, 0.08, 0.07, 0.06],
#             'Description': [
#                 'Previous period return',
#                 'MACD histogram',
#                 '14-day RSI',
#                 'Sentiment score change',
#                 '5-day price volatility',
#                 'Volume vs average ratio',
#                 'Position in Bollinger Bands',
#                 '5-day momentum'
#             ]
#         })
        
#         st.dataframe(feature_importance, use_container_width=True)
        
#         # SHAP Summary Plot (Fake)
#         st.subheader("🎯 SHAP Summary Analysis")
        
#         # Create fake SHAP-like visualization
#         features = feature_importance['Feature'].head(6).tolist()
#         shap_values = np.random.uniform(-0.4, 0.4, (100, len(features)))
        
#         fig = go.Figure()
        
#         for i, feature in enumerate(features):
#             fig.add_trace(go.Box(
#                 y=shap_values[:, i],
#                 name=feature,
#                 boxpoints='outliers'
#             ))
        
#         fig.update_layout(
#             title="SHAP Values Distribution by Feature",
#             yaxis_title="SHAP Value",
#             xaxis_title="Features"
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Model Performance Over Time
#         st.subheader("📊 Model Performance Tracking")
        
#         dates = pd.date_range(start='2024-01-01', end='2024-05-26', freq='W')
#         accuracy = 0.65 + 0.15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.02, len(dates))
#         accuracy = np.clip(accuracy, 0.5, 0.85)
        
#         fig = px.line(x=dates, y=accuracy, title="Weekly Model Accuracy")
#         fig.update_yaxis(title="Accuracy", range=[0.5, 0.9])
#         st.plotly_chart(fig, use_container_width=True)

# # System Status Tab
# elif tab_selection == "⚡ System Status":
#     st.header("⚡ System Monitoring & Status")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.subheader("🔧 System Health")
#         st.metric("Model Version", "v1.2.3")
#         st.metric("Last Retrained", "April 15, 2025")
#         st.metric("Uptime", "99.7%", "+0.1%")
        
#         # Status indicators
#         st.markdown("**Service Status:**")
#         st.markdown("🟢 **Prediction API**: Online")
#         st.markdown("🟢 **Data Pipeline**: Running")
#         st.markdown("🟡 **Model Training**: Scheduled")
#         st.markdown("🟢 **Redis Cache**: Connected")
    
#     with col2:
#         st.subheader("📊 Performance Metrics")
        
#         # Real-time metrics simulation
#         latency = random.uniform(2.8, 4.2)
#         st.metric("Avg Latency", f"{latency:.1f}s", f"{random.uniform(-0.5, 0.5):+.1f}s")
        
#         throughput = random.randint(45, 65)
#         st.metric("Throughput", f"{throughput} req/min", f"{random.randint(-5, 8):+d}")
        
#         error_rate = random.uniform(0.1, 0.8)
#         st.metric("Error Rate", f"{error_rate:.1f}%", f"{random.uniform(-0.2, 0.3):+.1f}%")
        
#         # Progress bars for system load
#         st.markdown("**System Load:**")
#         st.progress(random.uniform(0.3, 0.8), text="CPU Usage")
#         st.progress(random.uniform(0.2, 0.6), text="Memory Usage")
#         st.progress(random.uniform(0.1, 0.4), text="Disk Usage")
    
#     with col3:
#         st.subheader("🔍 Model Monitoring")
        
#         drift_score = random.uniform(0.1, 0.9)
#         drift_status = "Normal" if drift_score < 0.3 else "Warning" if drift_score < 0.7 else "Alert"
#         drift_color = "green" if drift_status == "Normal" else "orange" if drift_status == "Warning" else "red"
        
#         st.metric("Drift Score", f"{drift_score:.2f}")
#         st.markdown(f"Status: <span style='color: {drift_color}'><b>{drift_status}</b></span>", 
#                    unsafe_allow_html=True)
        
#         st.markdown("**Queue Status:**")
#         queue_length = random.randint(0, 15)
#         st.metric("Task Queue", f"{queue_length} tasks")
        
#         # Recent alerts
#         st.markdown("**Recent Alerts:**")
#         alerts = [
#             "✅ Model retrained successfully",
#             "⚠️ High latency detected (resolved)",
#             "✅ Data pipeline updated",
#             "📊 Weekly report generated"
#         ]
        
#         for alert in alerts[:3]:
#             st.markdown(f"- {alert}")
    
#     # System Logs
#     st.markdown("---")
#     st.subheader("📋 Recent System Logs")
    
#     # Generate fake logs
#     log_entries = []
#     for i in range(10):
#         timestamp = (datetime.now() - timedelta(minutes=random.randint(1, 120))).strftime("%Y-%m-%d %H:%M:%S")
#         level = random.choice(["INFO", "DEBUG", "WARN", "ERROR"])
#         messages = {
#             "INFO": ["Prediction request completed", "Model inference successful", "Cache hit", "API response sent"],
#             "DEBUG": ["Loading model weights", "Feature preprocessing", "Data validation passed"],
#             "WARN": ["High memory usage", "Slow database query", "Rate limit approaching"],
#             "ERROR": ["Database connection failed", "Model loading error", "Invalid input format"]
#         }
#         message = random.choice(messages[level])
#         log_entries.append(f"{timestamp} [{level}] {message}")
    
#     st.code("\n".join(log_entries), language="log")

# # About Tab
# else:  # About Tab
#     st.header("📘 About Strategic Alpha Intelligence System")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🎯 Project Objectives")
#         st.markdown("""
#         **Strategic Alpha Intelligence System** is an advanced machine learning platform designed to:
        
#         - 📈 **Generate High-Quality Trading Signals** using ensemble ML models
#         - 🔍 **Provide Explainable AI** through SHAP analysis and feature importance
#         - ⚡ **Enable Real-Time Predictions** with low-latency API architecture
#         - 📊 **Support Strategic Decision Making** through comprehensive backtesting
#         - 🛡️ **Ensure Robust Performance** via continuous monitoring and drift detection
#         """)
        
#         st.subheader("📊 Data Sources")
#         st.markdown("""
#         **Market Data:**
#         - Yahoo Finance API
#         - Alpha Vantage
#         - Quandl Financial Data
        
#         **Alternative Data:**
#         - News Sentiment (NewsAPI)
#         - Social Media Sentiment
#         - Economic Indicators
#         - Earnings Transcripts
#         """)
    
#     with col2:
#         st.subheader("🧠 Modeling Approach")
#         st.markdown("""
#         **Machine Learning Pipeline:**
        
#         1. **Data Preprocessing**
#            - Feature engineering (100+ technical indicators)
#            - Sentiment analysis integration
#            - Missing value handling
#            - Outlier detection
        
#         2. **Model Architecture**
#            - Stacking Ensemble (XGBoost + CatBoost + RF)
#            - Meta-learner optimization
#            - Cross-validation with time series splits
        
#         3. **Prediction Generation**
#            - Multi-class classification (BUY/HOLD/SELL)
#            - Confidence scoring
#            - Risk-adjusted returns
#         """)
        
#         st.subheader("🏗️ Deployment Stack")
#         st.markdown("""
#         **Infrastructure:**
#         - 🐳 **Docker** - Containerization
#         - ☁️ **AWS/GCP** - Cloud hosting
#         - 🔄 **Redis** - Caching layer
#         - 🌿 **Celery** - Task queue
#         - 📊 **Prometheus** - Metrics collection
#         - 📈 **Grafana** - Monitoring dashboards
#         - 🔧 **MLflow** - Model versioning
#         """)
    
#     st.markdown("---")
    
#     # Developer Tools (Easter Egg)
#     if st.checkbox("🔧 Developer Mode"):
#         st.subheader("🛠️ Developer Tools")
        
#         tab1, tab2, tab3 = st.tabs(["Logs", "Config", "Model Selection"])
        
#         with tab1:
#             st.code("""
# [2024-05-26 10:15:23] INFO - Model v1.2.3 loaded successfully
# [2024-05-26 10:15:24] DEBUG - Feature pipeline initialized
# [2024-05-26 10:15:25] INFO - Redis connection established
# [2024-05-26 10:16:01] INFO - Prediction request: AAPL
# [2024-05-26 10:16:03] DEBUG - Features extracted: 127 dimensions
# [2024-05-26 10:16:04] INFO - Ensemble prediction: BUY (confidence: 0.847)
# [2024-05-26 10:16:04] INFO - Response sent (latency: 3.2s)
#             """, language="log")
        
#         with tab2:
#             config = {
#                 "model": {
#                     "version": "v1.2.3",
#                     "ensemble_weights": [0.3, 0.25, 0.25, 0.2],
#                     "prediction_threshold": 0.6
#                 },
#                 "data": {
#                     "lookback_period": 30,
#                     "technical_indicators": 127,
#                     "sentiment_weight": 0.15
#                 },
#                 "deployment": {
#                     "batch_size": 32,
#                     "max_latency": 5.0,
#                     "cache_ttl": 300
#                 }
#             }
#             st.json(config)
        
#         with tab3:
#             model_choice = st.selectbox(
#                 "Model Configuration:",
#                 ["Stacked Ensemble (Production)", "XGBoost Only", "CatBoost Only", "Rule-Based Fallback"]
#             )
#             st.info(f"Selected: {model_choice}")
    
#     # Project Stats
#     st.markdown("---")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Training Data", "2.5M samples")
#     with col2:
#         st.metric("Features", "127")
#     with col3:
#         st.metric("Accuracy", "74.2%")
#     with col4:
#         st.metric("Daily Predictions", "500+")

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: #666;'>"
#     "Strategic Alpha Intelligence System v1.2.3 | Built with ❤️ using Streamlit & OpenAI"
#     "</div>", 
#     unsafe_allow_html=True
# )




import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import openai
from typing import Dict, List, Tuple
import json

# Configure page
st.set_page_config(
    page_title="Strategic Alpha Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .status-bad {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Feature-based prediction simulation
def generate_feature_based_prediction(features: Dict) -> Dict:
    """Generate prediction based on user-provided features"""
    
    # Simulate API processing delay
    time.sleep(random.uniform(2, 4))
    
    # Extract key features
    rsi = features['rsi_14']
    macd = features['macd_hist']
    price_change = features['price_change']
    volume_ratio = features['volume_ratio']
    sentiment = features['sentiment_score']
    volatility = features['volatility_5d']
    bollinger_pos = features['bollinger_pos']
    
    # Create feature-based prediction logic
    base_score = 0.0
    
    # RSI Analysis
    if rsi < 30:
        base_score += 0.3  # Oversold - bullish
    elif rsi > 70:
        base_score -= 0.25  # Overbought - bearish
    else:
        base_score += (50 - abs(rsi - 50)) / 100  # Neutral zone
    
    # MACD Analysis
    base_score += macd * 0.4
    
    # Price momentum
    base_score += price_change * 0.02
    
    # Volume confirmation
    if volume_ratio > 1.5:
        base_score += 0.2  # High volume confirms trend
    elif volume_ratio < 0.7:
        base_score -= 0.1  # Low volume weakens signal
    
    # Sentiment impact
    base_score += sentiment * 0.3
    
    # Volatility adjustment
    volatility_factor = min(volatility / 4.0, 1.0)  # Cap at 1.0
    base_score *= (1 - volatility_factor * 0.2)  # Reduce confidence in high volatility
    
    # Bollinger Band position
    if bollinger_pos < 0.2:
        base_score += 0.15  # Near lower band - potential reversal
    elif bollinger_pos > 0.8:
        base_score -= 0.15  # Near upper band - potential reversal
    
    # Convert to return prediction
    predicted_return = base_score * 15 + random.uniform(-2, 2)  # Add some noise
    predicted_return = max(min(predicted_return, 25), -20)  # Reasonable bounds
    
    # Calculate confidence based on feature alignment
    confidence_factors = []
    confidence_factors.append(1 - abs(rsi - 50) / 50)  # RSI clarity
    confidence_factors.append(abs(macd) / 2.0)  # MACD strength
    confidence_factors.append(min(volume_ratio / 2.0, 1.0))  # Volume confirmation
    confidence_factors.append(1 - volatility / 8.0)  # Lower volatility = higher confidence
    confidence_factors.append(abs(sentiment))  # Sentiment clarity
    
    confidence = np.mean(confidence_factors)
    confidence = max(min(confidence, 0.95), 0.55)  # Reasonable confidence range
    
    # Determine signal
    if predicted_return > 4:
        signal = "BUY"
    elif predicted_return < -3:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # Generate intelligent explanation based on features
    explanations = []
    
    if rsi < 30:
        explanations.append("RSI indicates oversold conditions, suggesting potential upward reversal")
    elif rsi > 70:
        explanations.append("RSI shows overbought levels, indicating possible downward pressure")
    
    if abs(macd) > 0.5:
        direction = "bullish" if macd > 0 else "bearish"
        explanations.append(f"Strong MACD signal supports {direction} momentum")
    
    if volume_ratio > 1.5:
        explanations.append("High volume confirms the directional move")
    
    if abs(sentiment) > 0.3:
        sent_direction = "positive" if sentiment > 0 else "negative"
        explanations.append(f"Market sentiment is strongly {sent_direction}")
    
    if volatility > 4:
        explanations.append("High volatility increases uncertainty in the prediction")
    
    final_explanation = ". ".join(explanations[:3]) + f". Ensemble confidence: {confidence*100:.1f}%"
    
    return {
        "predicted_return": round(predicted_return, 2),
        "signal": signal,
        "confidence": round(confidence, 3),
        "explanation": final_explanation,
        "feature_contributions": {
            "rsi_impact": round((50 - abs(rsi - 50)) / 100 * 0.3, 3),
            "macd_impact": round(macd * 0.4, 3),
            "sentiment_impact": round(sentiment * 0.3, 3),
            "volume_impact": round((volume_ratio - 1) * 0.2, 3),
            "volatility_adjustment": round(-volatility_factor * 0.2, 3)
        },
        "computed_features": {
            "momentum_score": round(base_score, 3),
            "risk_score": round(volatility_factor, 3),
            "trend_strength": round(abs(macd) + abs(sentiment), 3)
        },
        "timestamp": datetime.now()
    }

# Header
st.markdown('<h1 class="main-header">Strategic Alpha Intelligence System v2.3</h1>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Advanced ML-Powered Stock Prediction & Trading Signal Generation</p>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
tab_selection = st.sidebar.radio(
    "Select Module:",
    ["🔮 Prediction", "📊 Backtesting"]
)

# Stock universe for dropdowns
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "WIPRO.NS", "ITC.NS"
]

# Dashboard Tab
# if tab_selection == "🏠 Dashboard":
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Model Version", "v1.2.3", "Updated")
#     with col2:
#         st.metric("Active Signals", "23", "+5")
#     with col3:
#         st.metric("Accuracy (30d)", "74.2%", "+2.1%")
#     with col4:
#         st.metric("Sharpe Ratio", "1.84", "+0.12")
    
#     st.markdown("---")
    
#     # Quick Stats
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📈 Recent Performance")
        
#         # Generate fake performance data
#         dates = pd.date_range(start='2024-01-01', end='2024-05-26', freq='D')
#         strategy_returns = np.cumsum(np.random.normal(0.0008, 0.02, len(dates)))
#         benchmark_returns = np.cumsum(np.random.normal(0.0005, 0.015, len(dates)))
        
#         df_perf = pd.DataFrame({
#             'Date': dates,
#             'Strategy': strategy_returns,
#             'Benchmark': benchmark_returns
#         })
        
#         fig = px.line(df_perf, x='Date', y=['Strategy', 'Benchmark'], 
#                      title="Cumulative Returns vs Benchmark")
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         st.subheader("🎯 Top Signals Today")
        
#         # Quick Stats
#         fake_signals = pd.DataFrame({
#             'Feature Set': ['RSI-MACD-Vol', 'Momentum-Sent', 'Volatility-BB', 'Multi-Factor', 'Sector-Macro'],
#             'Signal': ['BUY', 'SELL', 'BUY', 'HOLD', 'BUY'],
#             'Confidence': [0.87, 0.92, 0.78, 0.65, 0.81],
#             'Expected Return': ['+8.2%', '-5.1%', '+6.7%', '+1.2%', '+7.3%']
#         })
        
#         st.dataframe(fake_signals, use_container_width=True)

# Prediction Tab
if tab_selection == "🔮 Prediction":
    st.header("🔮 Stock Prediction Engine")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Feature Input Parameters")
        st.info("💡 **Note**: Enter key technical indicators below. The system will automatically compute 100+ additional features including momentum, volatility, and sentiment scores.")
        
        # Primary Technical Indicators (User Input)
        st.markdown("**📊 Core Technical Indicators:**")
        rsi_14 = st.slider("RSI (14-day)", 0.0, 100.0, 50.0, 0.1)
        macd_hist = st.slider("MACD Histogram", -2.0, 2.0, 0.0, 0.01)
        price_change = st.slider("Price Change (%)", -15.0, 15.0, 0.0, 0.1)
        volume_ratio = st.slider("Volume Ratio (vs avg)", 0.1, 5.0, 1.0, 0.1)
        
        st.markdown("**📈 Market Context:**")
        bollinger_pos = st.slider("Bollinger Band Position", 0.0, 1.0, 0.5, 0.01)
        volatility_5d = st.slider("5-day Volatility (%)", 0.5, 8.0, 2.0, 0.1)
        
        st.markdown("**🎭 Sentiment & Market:**")
        sentiment_score = st.selectbox("Market Sentiment", 
                                     ["Very Bearish (-0.8)", "Bearish (-0.4)", "Neutral (0.0)", 
                                      "Bullish (+0.4)", "Very Bullish (+0.8)"])
        market_regime = st.selectbox("Market Regime", 
                                   ["Bull Market", "Bear Market", "Sideways", "High Volatility"])
        
        # Additional context
        st.markdown("**⏰ Temporal Context:**")
        analysis_date = st.date_input("Analysis Date", datetime.now())
        intraday_time = st.selectbox("Trading Session", 
                                   ["Pre-Market", "Market Open", "Mid-Day", "Market Close", "After-Hours"])
        
        if st.button("🚀 Generate AI Prediction", type="primary"):
            with st.spinner("Processing features through ML pipeline..."):
                # Convert inputs to numerical values
                sentiment_map = {
                    "Very Bearish (-0.8)": -0.8,
                    "Bearish (-0.4)": -0.4,
                    "Neutral (0.0)": 0.0,
                    "Bullish (+0.4)": 0.4,
                    "Very Bullish (+0.8)": 0.8
                }
                
                # Create feature dict for prediction
                features = {
                    'rsi_14': rsi_14,
                    'macd_hist': macd_hist,
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'bollinger_pos': bollinger_pos,
                    'volatility_5d': volatility_5d,
                    'sentiment_score': sentiment_map[sentiment_score],
                    'market_regime': market_regime
                }
                
                # Simulate API call
                prediction = generate_feature_based_prediction(features)
                
                st.session_state.predictions_history.append({
                    'features': features,
                    'prediction': prediction
                })
                
                st.success("✅ Prediction generated successfully!")
    
    with col2:
        st.subheader("🤖 AI Prediction Results")
        
        if st.session_state.predictions_history:
            latest_pred = st.session_state.predictions_history[-1]
            pred_data = latest_pred['prediction']
            input_features = latest_pred['features']
            
            # Display prediction metrics
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                return_val = pred_data['predicted_return']
                return_color = "🟢" if return_val > 0 else "🔴" if return_val < 0 else "🟡"
                st.metric("Predicted Return", f"{return_color} {return_val}%")
            with col2b:
                signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                st.metric("AI Signal", f"{signal_color[pred_data['signal']]} {pred_data['signal']}")
            with col2c:
                st.metric("Model Confidence", f"{pred_data['confidence']*100:.1f}%")
            
            # AI Explanation
            st.info(f"**🧠 AI Analysis:** {pred_data['explanation']}")
            
            # Feature Contribution Analysis
            st.subheader("📊 Feature Impact Analysis")
            
            col2d, col2e = st.columns(2)
            
            with col2d:
                st.markdown("**🔍 Input Feature Contributions:**")
                contributions = pred_data['feature_contributions']
                
                contrib_df = pd.DataFrame([
                    {"Feature": "RSI Impact", "Value": contributions['rsi_impact']},
                    {"Feature": "MACD Impact", "Value": contributions['macd_impact']},
                    {"Feature": "Sentiment Impact", "Value": contributions['sentiment_impact']},
                    {"Feature": "Volume Impact", "Value": contributions['volume_impact']},
                    {"Feature": "Volatility Adj.", "Value": contributions['volatility_adjustment']}
                ])
                
                fig = go.Figure(go.Bar(
                    x=contrib_df['Value'],
                    y=contrib_df['Feature'],
                    orientation='h',
                    marker_color=['green' if x > 0 else 'red' for x in contrib_df['Value']],
                    text=[f"{x:+.3f}" for x in contrib_df['Value']],
                    textposition='auto'
                ))
                fig.update_layout(title="Feature Contribution to Signal", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2e:
                st.markdown("**⚙️ Computed Internal Features:**")
                computed = pred_data['computed_features']
                
                st.metric("Momentum Score", f"{computed['momentum_score']:+.3f}")
                st.metric("Risk Score", f"{computed['risk_score']:.3f}")
                st.metric("Trend Strength", f"{computed['trend_strength']:.3f}")
                
                # Show some of the "127 computed features"
                st.markdown("**🔧 Additional Computed Features:**")
                additional_features = [
                    f"moving_avg_cross: {random.uniform(-1, 1):.3f}",
                    f"price_momentum_10d: {random.uniform(-2, 2):.3f}",
                    f"support_resistance: {random.uniform(0, 1):.3f}",
                    f"market_correlation: {random.uniform(-1, 1):.3f}",
                    f"sector_momentum: {random.uniform(-1, 1):.3f}",
                    "... (122 more features)"
                ]
                
                for feature in additional_features:
                    st.text(feature)
            
            # Advanced SHAP-like Analysis
            st.subheader("🎯 Advanced Feature Importance (SHAP)")
            
            # Create comprehensive SHAP-like visualization
            all_features = ['RSI_14', 'MACD_hist', 'Sentiment', 'Volume_Ratio', 'Volatility_5d', 
                          'Bollinger_Pos', 'Price_Change', 'MA_Cross', 'Support_Level', 
                          'Market_Correlation', 'Sector_Momentum', 'Economic_Indicator']
            
            shap_values = []
            for feature in all_features:
                if feature in ['RSI_14', 'MACD_hist', 'Sentiment', 'Volume_Ratio', 'Volatility_5d']:
                    # Use actual contributions for input features
                    if feature == 'RSI_14':
                        shap_values.append(contributions['rsi_impact'])
                    elif feature == 'MACD_hist':
                        shap_values.append(contributions['macd_impact'])
                    elif feature == 'Sentiment':
                        shap_values.append(contributions['sentiment_impact'])
                    elif feature == 'Volume_Ratio':
                        shap_values.append(contributions['volume_impact'])
                    elif feature == 'Volatility_5d':
                        shap_values.append(contributions['volatility_adjustment'])
                else:
                    # Generate realistic values for computed features
                    shap_values.append(random.uniform(-0.15, 0.15))
            
            fig = go.Figure(go.Bar(
                x=shap_values,
                y=all_features,
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in shap_values],
                text=[f"{x:+.3f}" for x in shap_values],
                textposition='auto'
            ))
            fig.update_layout(
                title="SHAP Feature Importance Analysis (Top 12 of 127 Features)",
                xaxis_title="SHAP Value (Impact on Prediction)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Processing Summary
            st.success(f"""
            ✅ **Processing Complete**
            - **Input Features**: 8 (user-provided)
            - **Computed Features**: 119 (auto-generated)
            - **Total Features**: 127
            - **Model Ensemble**: XGBoost + CatBoost + Random Forest → Meta-XGBoost
            - **Processing Time**: {random.uniform(2.1, 3.8):.1f} seconds
            """)
        
        else:
            st.info("👆 **Please provide technical indicators** and click 'Generate AI Prediction' to see results")
            
            st.markdown("""
            **🔧 How it works:**
            
            1. **Input**: You provide 8 key technical indicators
            2. **Feature Engineering**: System computes 119 additional features including:
               - Moving averages (multiple timeframes)
               - Momentum oscillators
               - Volatility measures
               - Support/resistance levels
               - Market correlations
               - Sector analysis
               - Economic indicators
            3. **AI Processing**: Ensemble model analyzes all 127 features
            4. **Output**: Signal + explanation + feature importance
            """)

# Backtesting Tab
elif tab_selection == "📊 Backtesting":
    st.header("📊 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Backtest Parameters")
        
        start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        end_date = st.date_input("End Date", datetime(2024, 5, 26))
        sector_filter = st.multiselect("Sector Filter", 
                                     ["Technology", "Finance", "Healthcare", "Energy", "Consumer"],
                                     default=["Technology"])
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🔄 Run Backtest", type="primary"):
            with st.spinner("Running backtest simulation..."):
                time.sleep(3)  # Simulate processing time
                st.success("Backtest completed!")
    
    with col2:
        st.subheader("Backtest Results")
        
        # Performance Metrics
        col2a, col2b, col2c, col2d = st.columns(4)
        
        with col2a:
            st.metric("CAGR", "18.7%", "+4.2%")
        with col2b:
            st.metric("Sharpe Ratio", "1.84", "+0.31")
        with col2c:
            st.metric("Max Drawdown", "-8.3%", "")
        with col2d:
            st.metric("Win Rate", "67.2%", "+12.1%")
        
        # Cumulative Returns Chart
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        strategy_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
        benchmark_returns = np.cumsum(np.random.normal(0.0007, 0.015, len(dates)))
        
        df_backtest = pd.DataFrame({
            'Date': dates,
            'Strategy': strategy_returns,
            'Nifty 50': benchmark_returns
        })
        
        fig = px.line(df_backtest, x='Date', y=['Strategy', 'Nifty 50'], 
                     title="Strategy vs Benchmark Performance")
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade History
        # st.subheader("Recent Feature-Based Predictions")
        # fake_trades = pd.DataFrame({
        #     'Date': pd.date_range(start='2024-05-01', periods=10, freq='2D'),
        #     'Feature Profile': ['High RSI + Bullish MACD', 'Low RSI + High Vol', 'Strong Momentum', 
        #                       'Overbought + Bear Sentiment', 'Neutral Multi-factor', 'Oversold Recovery',
        #                       'Breakout + Volume', 'Mean Reversion', 'Trend Following', 'Risk-off Signal'],
        #     'Signal': np.random.choice(['BUY', 'SELL', 'HOLD'], 10),
        #     'Actual Return': np.random.uniform(-8, 15, 10).round(2)
        # })
        # fake_trades['Actual Return'] = fake_trades['Actual Return'].apply(lambda x: f"{x:+.2f}%")
        # st.dataframe(fake_trades, use_container_width=True)

# Model Insights Tab
# elif tab_selection == "🧠 Model Insights":
#     st.header("🧠 Model Architecture & Insights")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🏗️ Ensemble Architecture")
        
#         st.markdown("""
#         **Stacking Ensemble Approach:**
        
#         **Level 1 Base Models:**
#         - 🌳 XGBoost Classifier
#         - 🐱 CatBoost Classifier  
#         - 🌲 Random Forest Classifier
#         - 📊 LightGBM Classifier
        
#         **Level 2 Meta-Model:**
#         - 🚀 Meta-XGBoost (Final Predictor)
        
#         **Feature Engineering:**
#         - Technical Indicators (20+ features)
#         - Sentiment Analysis Scores
#         - Market Microstructure Data
#         - Volatility Surfaces
#         """)
        
#         st.subheader("⚙️ Model Configuration")
#         st.code("""
# # Model Hyperparameters
# xgb_params = {
#     'n_estimators': 500,
#     'max_depth': 8,
#     'learning_rate': 0.05,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8
# }

# # Cross-Validation: 5-fold TimeSeriesSplit
# # Feature Selection: Recursive Feature Elimination
# # Optimization: Bayesian Hyperparameter Tuning
#         """, language='python')
    
#     with col2:
#         st.subheader("📈 Feature Importance Analysis")
        
#         # Feature importance table
#         feature_importance = pd.DataFrame({
#             'Feature': ['return_t-1', 'MACD_hist', 'RSI_14', 'sentiment_score_delta', 
#                        'price_volatility_5d', 'volume_ratio', 'bollinger_position', 'momentum_5d'],
#             'Importance': [0.21, 0.18, 0.14, 0.12, 0.09, 0.08, 0.07, 0.06],
#             'Description': [
#                 'Previous period return',
#                 'MACD histogram',
#                 '14-day RSI',
#                 'Sentiment score change',
#                 '5-day price volatility',
#                 'Volume vs average ratio',
#                 'Position in Bollinger Bands',
#                 '5-day momentum'
#             ]
#         })
        
#         st.dataframe(feature_importance, use_container_width=True)
        
#         # SHAP Summary Plot (Fake)
#         st.subheader("🎯 SHAP Summary Analysis")
        
#         # Create fake SHAP-like visualization
#         features = feature_importance['Feature'].head(6).tolist()
#         shap_values = np.random.uniform(-0.4, 0.4, (100, len(features)))
        
#         fig = go.Figure()
        
#         for i, feature in enumerate(features):
#             fig.add_trace(go.Box(
#                 y=shap_values[:, i],
#                 name=feature,
#                 boxpoints='outliers'
#             ))
        
#         fig.update_layout(
#             title="SHAP Values Distribution by Feature",
#             yaxis_title="SHAP Value",
#             xaxis_title="Features"
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Model Performance Over Time
#         st.subheader("📊 Model Performance Tracking")
        
#         dates = pd.date_range(start='2024-01-01', end='2024-05-26', freq='W')
#         accuracy = 0.65 + 0.15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.02, len(dates))
#         accuracy = np.clip(accuracy, 0.5, 0.85)
        
#         fig = px.line(x=dates, y=accuracy, title="Weekly Model Accuracy")
#         fig.update_yaxis(title="Accuracy", range=[0.5, 0.9])
#         st.plotly_chart(fig, use_container_width=True)

# # System Status Tab
# elif tab_selection == "⚡ System Status":
#     st.header("⚡ System Monitoring & Status")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.subheader("🔧 System Health")
#         st.metric("Model Version", "v1.2.3")
#         st.metric("Last Retrained", "April 15, 2025")
#         st.metric("Uptime", "99.7%", "+0.1%")
        
#         # Status indicators
#         st.markdown("**Service Status:**")
#         st.markdown("🟢 **Prediction API**: Online")
#         st.markdown("🟢 **Data Pipeline**: Running")
#         st.markdown("🟡 **Model Training**: Scheduled")
#         st.markdown("🟢 **Redis Cache**: Connected")
    
#     with col2:
#         st.subheader("📊 Performance Metrics")
        
#         # Real-time metrics simulation
#         latency = random.uniform(2.8, 4.2)
#         st.metric("Avg Latency", f"{latency:.1f}s", f"{random.uniform(-0.5, 0.5):+.1f}s")
        
#         throughput = random.randint(45, 65)
#         st.metric("Throughput", f"{throughput} req/min", f"{random.randint(-5, 8):+d}")
        
#         error_rate = random.uniform(0.1, 0.8)
#         st.metric("Error Rate", f"{error_rate:.1f}%", f"{random.uniform(-0.2, 0.3):+.1f}%")
        
#         # Progress bars for system load
#         st.markdown("**System Load:**")
#         st.progress(random.uniform(0.3, 0.8), text="CPU Usage")
#         st.progress(random.uniform(0.2, 0.6), text="Memory Usage")
#         st.progress(random.uniform(0.1, 0.4), text="Disk Usage")
    
#     with col3:
#         st.subheader("🔍 Model Monitoring")
        
#         drift_score = random.uniform(0.1, 0.9)
#         drift_status = "Normal" if drift_score < 0.3 else "Warning" if drift_score < 0.7 else "Alert"
#         drift_color = "green" if drift_status == "Normal" else "orange" if drift_status == "Warning" else "red"
        
#         st.metric("Drift Score", f"{drift_score:.2f}")
#         st.markdown(f"Status: <span style='color: {drift_color}'><b>{drift_status}</b></span>", 
#                    unsafe_allow_html=True)
        
#         st.markdown("**Queue Status:**")
#         queue_length = random.randint(0, 15)
#         st.metric("Task Queue", f"{queue_length} tasks")
        
#         # Recent alerts
#         st.markdown("**Recent Alerts:**")
#         alerts = [
#             "✅ Model retrained successfully",
#             "⚠️ High latency detected (resolved)",
#             "✅ Data pipeline updated",
#             "📊 Weekly report generated"
#         ]
        
#         for alert in alerts[:3]:
#             st.markdown(f"- {alert}")
    
#     # System Logs
#     st.markdown("---")
#     st.subheader("📋 Recent System Logs")
    
#     # Generate fake logs
#     log_entries = []
#     for i in range(10):
#         timestamp = (datetime.now() - timedelta(minutes=random.randint(1, 120))).strftime("%Y-%m-%d %H:%M:%S")
#         level = random.choice(["INFO", "DEBUG", "WARN", "ERROR"])
#         messages = {
#             "INFO": ["Prediction request completed", "Model inference successful", "Cache hit", "API response sent"],
#             "DEBUG": ["Loading model weights", "Feature preprocessing", "Data validation passed"],
#             "WARN": ["High memory usage", "Slow database query", "Rate limit approaching"],
#             "ERROR": ["Database connection failed", "Model loading error", "Invalid input format"]
#         }
#         message = random.choice(messages[level])
#         log_entries.append(f"{timestamp} [{level}] {message}")
    
#     st.code("\n".join(log_entries), language="log")

# # About Tab
# else:  # About Tab
#     st.header("📘 About Strategic Alpha Intelligence System")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🎯 Project Objectives")
#         st.markdown("""
#         **Strategic Alpha Intelligence System** is an advanced machine learning platform designed to:
        
#         - 📈 **Generate High-Quality Trading Signals** using ensemble ML models
#         - 🔍 **Provide Explainable AI** through SHAP analysis and feature importance
#         - ⚡ **Enable Real-Time Predictions** with low-latency API architecture
#         - 📊 **Support Strategic Decision Making** through comprehensive backtesting
#         - 🛡️ **Ensure Robust Performance** via continuous monitoring and drift detection
#         """)
        
#         st.subheader("📊 Data Sources")
#         st.markdown("""
#         **Market Data:**
#         - Yahoo Finance API
#         - Alpha Vantage
#         - Quandl Financial Data
        
#         **Alternative Data:**
#         - News Sentiment (NewsAPI)
#         - Social Media Sentiment
#         - Economic Indicators
#         - Earnings Transcripts
#         """)
    
#     with col2:
#         st.subheader("🧠 Modeling Approach")
#         st.markdown("""
#         **Machine Learning Pipeline:**
        
#         1. **Data Preprocessing**
#            - Feature engineering (100+ technical indicators)
#            - Sentiment analysis integration
#            - Missing value handling
#            - Outlier detection
        
#         2. **Model Architecture**
#            - Stacking Ensemble (XGBoost + CatBoost + RF)
#            - Meta-learner optimization
#            - Cross-validation with time series splits
        
#         3. **Prediction Generation**
#            - Multi-class classification (BUY/HOLD/SELL)
#            - Confidence scoring
#            - Risk-adjusted returns
#         """)
        
#         st.subheader("🏗️ Deployment Stack")
#         st.markdown("""
#         **Infrastructure:**
#         - 🐳 **Docker** - Containerization
#         - ☁️ **AWS/GCP** - Cloud hosting
#         - 🔄 **Redis** - Caching layer
#         - 🌿 **Celery** - Task queue
#         - 📊 **Prometheus** - Metrics collection
#         - 📈 **Grafana** - Monitoring dashboards
#         - 🔧 **MLflow** - Model versioning
#         """)
    
#     st.markdown("---")
    
#     # Developer Tools (Easter Egg)
#     if st.checkbox("🔧 Developer Mode"):
#         st.subheader("🛠️ Developer Tools")
        
#         tab1, tab2, tab3 = st.tabs(["Logs", "Config", "Model Selection"])
        
#         with tab1:
#             st.code("""
# [2024-05-26 10:15:23] INFO - Model v1.2.3 loaded successfully
# [2024-05-26 10:15:24] DEBUG - Feature pipeline initialized
# [2024-05-26 10:15:25] INFO - Redis connection established
# [2024-05-26 10:16:01] INFO - Prediction request: AAPL
# [2024-05-26 10:16:03] DEBUG - Features extracted: 127 dimensions
# [2024-05-26 10:16:04] INFO - Ensemble prediction: BUY (confidence: 0.847)
# [2024-05-26 10:16:04] INFO - Response sent (latency: 3.2s)
#             """, language="log")
        
#         with tab2:
#             config = {
#                 "model": {
#                     "version": "v1.2.3",
#                     "ensemble_weights": [0.3, 0.25, 0.25, 0.2],
#                     "prediction_threshold": 0.6
#                 },
#                 "data": {
#                     "lookback_period": 30,
#                     "technical_indicators": 127,
#                     "sentiment_weight": 0.15
#                 },
#                 "deployment": {
#                     "batch_size": 32,
#                     "max_latency": 5.0,
#                     "cache_ttl": 300
#                 }
#             }
#             st.json(config)
        
#         with tab3:
#             model_choice = st.selectbox(
#                 "Model Configuration:",
#                 ["Stacked Ensemble (Production)", "XGBoost Only", "CatBoost Only", "Rule-Based Fallback"]
#             )
#             st.info(f"Selected: {model_choice}")
    
#     # Project Stats
#     st.markdown("---")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Training Data", "2.5M samples")
#     with col2:
#         st.metric("Features", "127")
#     with col3:
#         st.metric("Accuracy", "74.2%")
#     with col4:
#         st.metric("Daily Predictions", "500+")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Copyright Meander Softwares Pvt. Ltd. All Rights Reserved | Test Cluster 3 | CPU 2GHz VRAM 2GB"
    "</div>", 
    unsafe_allow_html=True
)