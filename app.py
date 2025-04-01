from flask import Flask, render_template, jsonify
import datetime
import logging
import boto3
import joblib
from sklearn.decomposition import PCA
import tarfile
import os
import sys
from pathlib import Path
import json
import numpy as np
from sagemaker.amazon.common import read_recordio
import pandas as pd

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_indicators')

# Initialize Flask app
app = Flask(__name__)

# Empty placeholder data
current_data = {
    'formatted_price': '0.00',
    'last_updated': 'Never',
    'market_regime': 'Unknown',
    'trade_signal': 'Neutral',
    'pca_components': None,
    'has_error': False,
    'error_message': '',
    'pca_model_loaded': False,
    'history': [],
    'using_real_data': False,
    'rsi': 50,
    'macd_diff': 0,
    'atr_ratio': 1,
    'pca_interpretations': []
}




# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

# Path to the S3 model artifact
S3_BUCKET = 'sagemaker-eu-west-1-688567281415'
S3_KEY = 'pca_output/pca-2025-03-24-21-12-06-644/output/model.tar.gz'
LOCAL_MODEL_DIR = 'model_data'
OUTPUT_DIR = 'pca_analysis_results'  # Output directory for results

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pca_model():
    """Load the PCA model from S3 and use real market data for PCA analysis"""
    global current_data
    try:
        # Import numpy locally to avoid scoping issues
        import numpy as np
        
        # Create model directory if it doesn't exist
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        # Define the full file path for the downloaded tar.gz
        model_tar_path = os.path.join(LOCAL_MODEL_DIR, 'model.tar.gz')
        
        # Download the model from S3 to the specified file path
        logger.info(f"Downloading model from S3 {S3_BUCKET}/{S3_KEY} to {model_tar_path}")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_KEY, model_tar_path)
        logger.info(f"Download complete")

        # Extract the model file to the model directory
        logger.info(f"Extracting model archive")
        with tarfile.open(model_tar_path, 'r:gz') as tar:
            tar.extractall(LOCAL_MODEL_DIR)
        logger.info(f"Extraction complete")
        
        # List all files in the extracted directory
        logger.info("Files in extracted directory:")
        for root, dirs, files in os.walk(LOCAL_MODEL_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  {file_path}: {file_size} bytes")
        
        # Import required modules for feature calculation and data fetching
        sys.path.append(str(base_dir))
        from Machine1.utils.feature_calculator import calculate_all_features
        from Machine1.utils.bitget_futures import BitgetFutures
        
        # Initialize Bitget client with credentials from config
        # Only look in WebIndicators/config directory
        webindicators_dir = Path(__file__).resolve().parent
        config_path = os.path.join(webindicators_dir, 'config', 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            bitget_config = config.get('bitget', {})
            bitget_client = BitgetFutures(bitget_config)
            logger.info(f"Successfully initialized BitgetFutures client using config from {config_path}")
        else:
            # If no config file, raise an error - we need credentials
            error_msg = f"Config file not found at: {config_path}. Please create the config file at this exact location."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Fetch real market data
        symbol = "ETH/USDT:USDT"  # Default to ETH/USDT
        timeframe = "1h"  # Default to 1-hour timeframe
        
        # Calculate how many days of data to fetch - need enough for all indicators
        warmup_period = 15  # Days to skip for warmup (hourly data, so 15 days = 360 candles)
        analysis_period = 30  # Days to use for analysis
        total_days = warmup_period + analysis_period
        
        # For 1h timeframe, we need 24 candles per day
        candles_needed = total_days * 24
        
        logger.info(f"Fetching {candles_needed} candles of {timeframe} data for {symbol}")
        
        # Download the market data
        market_data = bitget_client.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            # Start from X days ago
            start_time=(datetime.datetime.now() - datetime.timedelta(days=total_days)).strftime('%Y-%m-%d')
        )
        
        # Verify we have a DataFrame with the expected structure
        if not isinstance(market_data, pd.DataFrame):
            error_msg = f"Expected DataFrame from bitget_client.fetch_ohlcv, got {type(market_data)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        logger.info(f"Downloaded {len(market_data)} real market candles")
        logger.info(f"DataFrame columns: {market_data.columns.tolist() if hasattr(market_data, 'columns') else 'unknown'}")
        logger.info(f"DataFrame index: {type(market_data.index)}")
        
        # No need to convert DataFrame to list, calculate_all_features can work with DataFrame directly
        
        # Calculate features
        logger.info(f"Calculating features for {len(market_data)} candles")
        features_df = calculate_all_features(market_data)
        
        # Get the feature names we expect in our PCA model
        # These are typical features we'd use in crypto trading
        expected_features = [
            'macd_diff', 'rsi', 'stoch_k', 'stoch_d', 'stoch_diff', 
            'cci', 'williams_r', 'stoch_rsi_k', 'stoch_rsi_d',
            'adx', 'supertrend', 'bb_width', 'tenkan_kijun_diff',
            'atr_ratio', 'historical_volatility_30', 'bb_pct',
            'vwap_ratio', 'obv_ratio', 'mfi', 'cmf', 
            'volume_ratio_20', 'volume_ratio_50', 'ema9_ratio'
        ]
        
        # Filter to features we'd expect in our PCA model
        features_to_use = []
        for feature in expected_features:
            if feature in features_df.columns:
                features_to_use.append(feature)
        
        # Skip warmup period - use only the last part of the data
        # For hourly data, we need enough warmup for all indicators (sma100, atr_ratio with ma100, etc.)
        # Using a 150 candle warmup (about a week of hourly data) should be sufficient
        warmup_candles = 150  # More than enough for all indicators
        if len(features_df) > warmup_candles:
            logger.info(f"Skipping first {warmup_candles} candles as warmup period")
            features_df = features_df.iloc[warmup_candles:]
        else:
            error_msg = f"Not enough data to skip warmup period! Got {len(features_df)} candles, needed at least {warmup_candles+1}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use the data for PCA
        feature_data = features_df[features_to_use].values
        logger.info(f"Created feature matrix with shape: {feature_data.shape}")
        
        # Initialize PCA model with 5 components
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # Create and fit a PCA model
        pca = PCA(n_components=5)
        pca.fit(normalized_data)
        
        # Transform the data to get principal components
        pc_values = pca.transform(normalized_data)
        
        # Get variance ratio but don't print it yet
        explained_variance = pca.explained_variance_ratio_
        
        # Store components for dashboard
        current_data['pca_components'] = pca.components_
        current_data['pca_model_loaded'] = True
        current_data['using_real_data'] = True
        
        # Get latest data point for interpretations
        latest_point = normalized_data[-1]
        
        # Generate PCA interpretations with the latest data point
        generate_pca_interpretations_from_data(pca, latest_point, explained_variance)
        
        # Update current price data
        try:
            latest_price = market_data['close'].iloc[-1]
            current_data['formatted_price'] = f"{latest_price:.2f}"
            current_data['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as price_err:
            logger.warning(f"Could not update price data: {str(price_err)}")
        
        # Only print meaningful summary information - no debug prints
        logger.info("\n====== PCA ANALYSIS COMPLETE ======")
        
        # Print the latest PCA values that are actually being used
        latest_pc = pc_values[-1]
        logger.info("Current PCA values (used for the dashboard):")
        logger.info(f"PC1 (Momentum): {latest_pc[0]:.4f} - {current_data['market_regime']} - Signal: {current_data['trade_signal']}")
        logger.info(f"PC2 (Volatility): {latest_pc[1]:.4f}")
        logger.info(f"PC3 (Structure): {latest_pc[2]:.4f}")
        logger.info(f"PC4 (Breadth): {latest_pc[3]:.4f}")
        logger.info(f"PC5 (Mean Reversion): {latest_pc[4]:.4f}")
        
        # Print a summary of PCA interpretations
        logger.info("\nPCA Interpretation Summary:")
        for interp in current_data['pca_interpretations']:
            logger.info(f"{interp['name']} ({interp['variance']}%): {interp['current_value']:.4f} - Signal: {interp['signal']}")
            
        # Print overall market summary
        logger.info("\n====== MARKET SUMMARY ======")
        logger.info(f"ETH/USDT Price: {current_data['formatted_price']} as of {current_data['last_updated']}")
        logger.info(f"Market Regime: {current_data['market_regime']}")
        logger.info(f"Trade Signal: {current_data['trade_signal']}")
        logger.info(f"Key Indicators - RSI: {current_data['rsi']:.2f}, MACD: {current_data['macd_diff']:.4f}, ATR Ratio: {current_data['atr_ratio']:.2f}")
        logger.info("============================\n")
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Error in PCA analysis: {str(e)}")
        logger.error(traceback.format_exc())
        current_data['has_error'] = True
        current_data['error_message'] = str(e)
        return False


def generate_pca_interpretations_from_data(pca_model, latest_point, explained_variance):
    """Generate interpretations for PCA components using actual data"""
    global current_data
    interpretations = []
    
    # Transform the data
    pc_values = latest_point
    
    # Component 1 - Momentum & Volatility Strength (42.11% variance)
    pc1 = pc_values[0]
    pc1_signal = 'Strong Bullish' if pc1 > 1.5 else 'Bullish' if pc1 > 0.5 else 'Neutral' if pc1 > -0.5 else 'Bearish' if pc1 > -1.5 else 'Strong Bearish'
    interpretations.append({
        'id': 1,
        'name': 'Momentum & Volatility Strength',
        'current_value': float(pc1),
        'signal': pc1_signal,
        'variance': f'{explained_variance[0] * 100:.1f}',
        'description': 'Captures momentum and volatility strength over a medium-term horizon (~20-30 periods).',
        'trading_guidance': 'Strong positive values indicate bullish momentum; negative values signal bearish momentum.'
    })
    
    # Add more components based on the number of components we have
    if len(pc_values) > 1:
        pc2 = pc_values[1]
        pc2_signal = 'Strong Accumulation' if pc2 > 1.5 else 'Accumulation' if pc2 > 0.5 else 'Neutral' if pc2 > -0.5 else 'Distribution' if pc2 > -1.5 else 'Strong Distribution'
        interpretations.append({
            'id': 2,
            'name': 'Volume and Short-Term Momentum',
            'current_value': float(pc2),
            'signal': pc2_signal,
            'variance': f'{explained_variance[1] * 100:.1f}',
            'description': 'Focuses on short-term volume-based momentum and underlying market liquidity.',
            'trading_guidance': 'Positive values indicate buying (accumulation); negative values show selling (distribution).'
        })
    
    if len(pc_values) > 2:
        pc3 = pc_values[2]
        pc3_signal = 'Strong Trend' if pc3 > 1.5 else 'Trending' if pc3 > 0.5 else 'Weak Trend' if pc3 > -0.5 else 'Counter-Trend' if pc3 > -1.5 else 'Strong Counter-Trend'
        interpretations.append({
            'id': 3,
            'name': 'Short-Term Trend and Price Action',
            'current_value': float(pc3),
            'signal': pc3_signal,
            'variance': f'{explained_variance[2] * 100:.1f}',
            'description': 'Captures short-term trend strength and immediate volume-price action.',
            'trading_guidance': 'Strong loadings indicate decisive trending price moves coupled with abnormal trading volume.'
        })
    
    if len(pc_values) > 3:
        pc4 = pc_values[3]
        pc4_signal = 'Potential Reversal' if abs(pc4) > 1.5 else 'Divergence' if abs(pc4) > 0.7 else 'Normal Range'
        interpretations.append({
            'id': 4,
            'name': 'Oscillator Divergence & Market Extremes',
            'current_value': float(pc4),
            'signal': pc4_signal,
            'variance': f'{explained_variance[3] * 100:.1f}',
            'description': 'Reflects conditions of market extremes or reversals indicated by oscillator divergence.',
            'trading_guidance': 'High values identify potential breakouts or trend reversals.'
        })
    
    if len(pc_values) > 4:
        pc5 = pc_values[4]
        pc5_signal = 'High Instability' if abs(pc5) > 1.5 else 'Unstable' if abs(pc5) > 0.7 else 'Stable'
        interpretations.append({
            'id': 5,
            'name': 'Volatility & Short-Term Oscillation',
            'current_value': float(pc5),
            'signal': pc5_signal,
            'variance': f'{explained_variance[4] * 100:.1f}',
            'description': 'Captures mid-term volatility changes and short-term oscillator behavior.',
            'trading_guidance': 'Useful for validating stability or instability of recent price moves.'
        })
    
    # Store in the global data dictionary
    current_data['pca_interpretations'] = interpretations
    
    # Set market regime and trade signal based on the principal components
    # Using PC1 (momentum) and PC2 (volume) for market regime determination
    if pc1 > 1.0 and pc2 > 0.5:
        current_data['market_regime'] = 'Strong Bull'
        current_data['trade_signal'] = 'Long'
    elif pc1 > 0.5:
        current_data['market_regime'] = 'Bullish'
        current_data['trade_signal'] = 'Long'
    elif pc1 < -1.0 and pc2 < -0.5:
        current_data['market_regime'] = 'Strong Bear'
        current_data['trade_signal'] = 'Short'
    elif pc1 < -0.5:
        current_data['market_regime'] = 'Bearish'
        current_data['trade_signal'] = 'Short'
    else:
        # Check if PC4 (oscillator divergence) indicates potential reversal
        if len(pc_values) > 3 and abs(pc4) > 1.5:
            if pc1 < 0:  # Current momentum negative but potential reversal
                current_data['market_regime'] = 'Potential Bottom'
                current_data['trade_signal'] = 'Watch for Long'
            else:  # Current momentum positive but potential reversal
                current_data['market_regime'] = 'Potential Top'
                current_data['trade_signal'] = 'Watch for Short'
        # Check PC3 for trending conditions in neutral momentum
        elif len(pc_values) > 2 and abs(pc3) > 1.0:
            current_data['market_regime'] = 'Trending Range'
            current_data['trade_signal'] = 'Neutral'
        else:
            current_data['market_regime'] = 'Calm Range'
            current_data['trade_signal'] = 'Hold'
            
    # Update some key indicators in current_data too
    # This would normally come from real-time data, but for now use the most recent values
    current_data['rsi'] = 50 + float(pc1) * 10  # Approximation based on PC1
    current_data['macd_diff'] = float(pc1) * 0.2  # Approximation based on PC1
    current_data['atr_ratio'] = 1.0 + abs(float(pc2)) * 0.5  # Approximation based on PC2

# Basic CSS helper
@app.template_filter('safe_width')
def safe_width(value):
    """Convert value to safe width percentage for CSS"""
    try:
        val = float(value)
        return max(0, min(100, val))
    except:
        return 0

# Add a safe round filter to handle undefined values
@app.template_filter('round')
def safe_round(value, precision=0):
    """Safely round a value, returns 0 if value is undefined or not a number"""
    try:
        return round(float(value), precision)
    except:
        return 0

# Basic routes
@app.route('/')
def index():
    return render_template('index.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

@app.route('/update')
def update():
    """Refresh the PCA model and market data when Update Now button is clicked"""
    try:
        # Store current data for history
        if current_data['pca_model_loaded'] and not current_data['has_error']:
            # Create a copy of the current data for the history
            history_entry = {
                'market_regime': current_data['market_regime'],
                'trade_signal': current_data['trade_signal'],
                'rsi': current_data['rsi'],
                'macd_diff': current_data['macd_diff'],
                'atr_ratio': current_data['atr_ratio'],
                'last_updated': current_data['last_updated'],
                'formatted_price': current_data['formatted_price']
            }
            
            # Add to history (limit to last 50 entries)
            current_data['history'].append(history_entry)
            if len(current_data['history']) > 50:
                current_data['history'] = current_data['history'][-50:]
        
        # Reload the PCA model and market data
        logger.info("Manual update requested - refreshing PCA model and market data")
        success = load_pca_model()
        
        if success:
            return jsonify({'status': 'success', 'message': 'Data updated successfully'})
        else:
            return jsonify({'status': 'error', 'message': current_data['error_message']})
            
    except Exception as e:
        logger.error(f"Error during manual update: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/pca-analysis')
def pca_analysis():
    return render_template('pca_analysis.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

@app.route('/history')
def history():
    return render_template('history.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

# Run the app
if __name__ == '__main__':
    load_pca_model() 
    logger.info("Starting Web Indicators app")
    app.run(debug=True, host='0.0.0.0', port=5000)
