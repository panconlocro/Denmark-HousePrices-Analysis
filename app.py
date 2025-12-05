"""
🏠 Denmark Housing Price Predictor
Aplicación Streamlit para predecir precios de propiedades en Dinamarca
usando múltiples modelos de Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Denmark House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        color: #333;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #1f77b4;
        font-weight: 700;
    }
    .metric-card p {
        color: #666;
        font-size: 0.9rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Carga los modelos entrenados"""
    models = {}
    try:
        models['XGBoost'] = joblib.load('models/xgb_cpu.pkl')
        models['RandomForest'] = joblib.load('models/rf.pkl')
        models['Ridge'] = joblib.load('models/ridge.pkl')
        models['FLAML AutoML'] = joblib.load('models/flaml_automl.pkl')
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
    return models


@st.cache_resource
def load_train_data():
    """Carga datos de entrenamiento para el scaler"""
    try:
        train = pd.read_parquet('data/processed/train_data.parquet')
        selected_features = open('data/processed/selected_features.txt').read().splitlines()
        
        exclude_features = ['log_price', 'quarter', 'region_count', 'price_deviation_from_median', 
                           'time_trend', 'region_target_encoded', 'region_count']
        features = [f for f in selected_features if f not in exclude_features]
        
        return train, features
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None


def create_features(date, region, house_type, sales_type, sqm, no_rooms, year_build):
    """
    Genera todas las features necesarias a partir de los inputs del usuario
    """
    # Fecha y temporales
    dt = pd.to_datetime(date)
    year = dt.year
    month = dt.month
    quarter = (month - 1) // 3 + 1
    
    # Features temporales
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    
    # Features de tamaño
    price_per_sqm = 0  # Placeholder, se calcula después
    rooms_sqm_ratio = no_rooms / sqm if sqm > 0 else 0
    
    # Edad de la propiedad
    property_age = year - year_build
    
    # Features categóricas (se codificarán como one-hot)
    is_premium = 1 if house_type == 'Villa' else 0
    
    # Interacciones básicas
    age_x_villa = property_age * is_premium
    
    # Features geográficas sintéticas (simplificadas)
    # En producción, estos valores vendrían de un mapeo region -> características
    region_frequency = 0  # Se calculará del dataset
    region_price_mean = 0  # Se calculará del dataset
    sqm_x_region = sqm * hash(region) % 100 / 100  # Proxy simple
    price_per_sqm_x_region = 0  # Se calculará después
    
    # Categorías de precio (se asignarán después de la predicción inicial)
    price_category_Premium = 0
    price_category_High = 0
    price_category_Medium = 0
    
    # Sales type
    sales_type_regular_sale = 1 if sales_type == 'Regular Sale' else 0
    
    # Crear DataFrame con todas las features
    features = {
        'year': year,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'quarter_sin': quarter_sin,
        'quarter_cos': quarter_cos,
        'sqm': sqm,
        'no_rooms': no_rooms,
        'property_age': property_age,
        'price_per_sqm': price_per_sqm,
        'rooms_sqm_ratio': rooms_sqm_ratio,
        'is_premium': is_premium,
        'age_x_villa': age_x_villa,
        'region_frequency': region_frequency,
        'region_price_mean': region_price_mean,
        'sqm_x_region': sqm_x_region,
        'price_per_sqm_x_region': price_per_sqm_x_region,
        'price_category_Premium': price_category_Premium,
        'price_category_High': price_category_High,
        'price_category_Medium': price_category_Medium,
        'sales_type_regular_sale': sales_type_regular_sale
    }
    
    return pd.DataFrame([features])


def predict_price(models, features_df, scaler):
    """
    Realiza predicciones con todos los modelos
    """
    # Escalar features
    features_scaled = pd.DataFrame(
        scaler.transform(features_df), 
        columns=features_df.columns
    )
    
    predictions = {}
    for name, model in models.items():
        try:
            log_pred = model.predict(features_scaled)[0]
            price_pred = np.exp(log_pred)
            predictions[name] = {
                'log_price': log_pred,
                'price_dkk': price_pred,
                'price_millions': price_pred / 1e6
            }
        except Exception as e:
            st.warning(f"Error en {name}: {e}")
    
    return predictions


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">Denmark Housing Price Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta aplicación predice el precio de propiedades en Dinamarca usando **4 modelos** de Machine Learning:
    - **XGBoost** (Gradient Boosting optimizado)
    - **RandomForest** (Ensamble de árboles de decisión)
    - **Ridge Regression** (Modelo lineal con regularización L2)
    - **FLAML AutoML** (Optimización automática de modelos)
    """)
    
    # Cargar modelos y datos
    with st.spinner('Cargando modelos...'):
        models = load_models()
        train_data, feature_names = load_train_data()
    
    if not models or train_data is None:
        st.error("No se pudieron cargar los modelos o datos necesarios.")
        return
    
    # Preparar scaler
    exclude_features = ['log_price', 'quarter', 'region_count', 'price_deviation_from_median', 
                       'time_trend', 'region_target_encoded', 'region_count']
    features = [f for f in feature_names if f not in exclude_features]
    X_train = train_data[features]
    scaler = StandardScaler().fit(X_train)
    
    st.success(f"{len(models)} modelos cargados correctamente")
    
    # ========================================================================
    # SIDEBAR - INPUTS DEL USUARIO
    # ========================================================================
    
    st.sidebar.header("Información de la Propiedad")
    
    # Fecha
    date = st.sidebar.date_input(
        "Fecha de venta",
        value=datetime.now(),
        min_value=datetime(2010, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    # Región
    regions = ['Capital Region', 'Zealand', 'Southern Denmark', 'Central Jutland', 'North Jutland']
    region = st.sidebar.selectbox("Región", regions)
    
    # Tipo de casa
    house_types = ['Apartment', 'Terraced house', 'Villa', 'Semi-detached house']
    house_type = st.sidebar.selectbox("Tipo de propiedad", house_types)
    
    # Tipo de venta
    sales_types = ['Regular Sale', 'Foreclosure', 'Other']
    sales_type = st.sidebar.selectbox("Tipo de venta", sales_types)
    
    # Características numéricas
    st.sidebar.subheader("Características")
    sqm = st.sidebar.number_input("Área (m²)", min_value=10, max_value=1000, value=100, step=10)
    no_rooms = st.sidebar.number_input("Número de habitaciones", min_value=1, max_value=20, value=3, step=1)
    year_build = st.sidebar.number_input("Año de construcción", min_value=1800, max_value=2024, value=1990, step=1)
    
    # Botón de predicción
    predict_button = st.sidebar.button("Predecir Precio", type="primary", use_container_width=True)
    
    # ========================================================================
    # ÁREA PRINCIPAL - RESULTADOS
    # ========================================================================
    
    if predict_button:
        with st.spinner('Generando predicciones...'):
            # Generar features
            features_df = create_features(date, region, house_type, sales_type, sqm, no_rooms, year_build)
            
            # Alinear features con el modelo
            missing_features = set(features) - set(features_df.columns)
            for feat in missing_features:
                features_df[feat] = 0
            features_df = features_df[features]
            
            # Hacer predicciones
            predictions = predict_price(models, features_df, scaler)
        
        # Mostrar resumen de la propiedad
        st.subheader("Resumen de la Propiedad")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Área", f"{sqm} m²")
            st.metric("Habitaciones", no_rooms)
        with col2:
            st.metric("Tipo", house_type)
            st.metric("Región", region)
        with col3:
            st.metric("Año construcción", year_build)
            st.metric("Edad", date.year - year_build)
        with col4:
            st.metric("Tipo de venta", sales_type)
            st.metric("Precio/m²", f"~{predictions['RandomForest']['price_dkk']/sqm:,.0f} DKK")
        
        st.markdown("---")
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3 = st.tabs(["Predicciones", "Comparación", "Detalles Técnicos"])
        
        with tab1:
            st.subheader("Predicciones por Modelo")
            
            # Calcular predicción promedio
            avg_price = np.mean([p['price_dkk'] for p in predictions.values()])
            
            # Mostrar predicción destacada
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Precio Estimado Promedio</h2>
                <h1>{avg_price:,.0f} DKK</h1>
                <h3>≈ {avg_price/1e6:.2f} millones DKK</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Predicciones individuales
            cols = st.columns(4)
            for idx, (name, pred) in enumerate(predictions.items()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{name}</h3>
                        <h2>{pred['price_dkk']:,.0f} DKK</h2>
                        <p>{pred['price_millions']:.2f}M DKK</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Comparación entre Modelos")
            
            # Crear gráfico de barras
            model_names = list(predictions.keys())
            prices = [predictions[name]['price_dkk'] for name in model_names]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=model_names,
                    y=prices,
                    text=[f'{p/1e3:.0f}k' for p in prices],
                    textposition='auto',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                )
            ])
            
            fig.update_layout(
                title="Predicciones por Modelo (DKK)",
                xaxis_title="Modelo",
                yaxis_title="Precio (DKK)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precio Mínimo", f"{min(prices):,.0f} DKK")
            with col2:
                st.metric("Precio Máximo", f"{max(prices):,.0f} DKK")
            with col3:
                st.metric("Desviación Estándar", f"{np.std(prices):,.0f} DKK")
        
        with tab3:
            st.subheader("Detalles Técnicos")
            
            # Mostrar features generadas
            st.write("**Features generadas para la predicción:**")
            features_display = features_df.copy()
            features_display = features_display.T
            features_display.columns = ['Valor']
            st.dataframe(features_display, use_container_width=True)
            
            # Información de los modelos
            st.markdown("""
            **Información de los Modelos:**
            
            - **RandomForest**: Ensamble de 400 árboles, `max_depth=None`, `min_samples_leaf=2`
            - **XGBoost**: 300 árboles, `learning_rate=0.05`, `tree_method=hist` (CPU optimizado)
            - **Ridge**: Regresión lineal con regularización L2, `alpha=10.0`
            - **FLAML AutoML**: Optimización automática con presupuesto de 600s
            
            **Target**: `log_price` (logaritmo del precio de compra)
            
            **Métricas en test set (613k muestras):**
            - RandomForest RMSE: 38,812 DKK
            - XGBoost RMSE: ~81,000 DKK
            - FLAML RMSE: ~64,000 DKK
            """)
    
    else:
        st.info("Completa los datos en el panel lateral y presiona **Predecir Precio**")
        
        # Mostrar estadísticas del dataset
        st.subheader("Estadísticas del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Muestras de Entrenamiento", f"{len(train_data):,}")
        with col2:
            st.metric("Features", len(features))
        with col3:
            st.metric("Periodo", "1992-2017")
        with col4:
            st.metric("Regiones", "5")


if __name__ == "__main__":
    main()
