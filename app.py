from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
import sys
import numpy as np
import traceback

# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# Import MarketConditionInterpreter and BitgetFutures
from Machine1.strategy.market_interpreter import MarketConditionInterpreter
from Machine1.utils.bitget_futures import BitgetFutures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_flask.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Custom filter for safe absolute value and percent calculation
@app.template_filter('safe_width')
def safe_width_filter(value, max_width=100, scale_factor=3):
    """Convert a value to a safe width percentage for progress bars"""
    try:
        # Convert to float if it's a numpy type
        if isinstance(value, np.number):
            value = float(value)
        elif not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return 0
        # Get absolute value
        abs_value = abs(value)
        # Calculate percentage with scaling
        percent = min(abs_value / scale_factor * 100, max_width)
        return percent
    except (ValueError, TypeError) as e:
        logging.error(f"Error in safe_width filter: {str(e)}")
        return 0

# Global variable to store market conditions
market_data = {
    'market_regime': 'Loading...',
    'trade_signal': 'Loading...',
    'pc1': 0.0,
    'pc2': 0.0,
    'pc3': 0.0,
    'price': 0.0,
    'rsi': 0.0,
    'adx': 0.0,
    'macd_diff': 0.0,
    'bb_width': 0.0,
    'last_updated': 'Never',
    'history': []  # To store historical data
}

# Initialize the market interpreter and BitGet client
interpreter = None
bitget_client = None

# Function to convert NumPy values to Python native types
def convert_numpy_values(data):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(data, dict):
        return {key: convert_numpy_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_values(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return convert_numpy_values(data.tolist())
    elif isinstance(data, datetime):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return data

def initialize_clients():
    """Initialize the market interpreter and BitGet client"""
    global interpreter, bitget_client
    try:
        # Initialize MarketConditionInterpreter
        if interpreter is None:
            logging.info("Initializing MarketConditionInterpreter...")
            # Force a call to the direct main method to ensure proper initialization
            try:
                logging.info("Forcing a direct initialization with full historical data...")
                temp_interpreter = MarketConditionInterpreter()
                
                # Log the means from the temp instance to verify proper initialization
                pc1_mean = getattr(temp_interpreter, 'pc1_mean', 0.0)
                pc2_mean = getattr(temp_interpreter, 'pc2_mean', 0.0)
                pc3_mean = getattr(temp_interpreter, 'pc3_mean', 0.0)
                
                if abs(pc1_mean) < 0.01 and abs(pc2_mean) < 0.01 and abs(pc3_mean) < 0.01:
                    logging.warning("ALERT: PC means are close to zero in temp instance, which is likely incorrect")
                    # Try to explicitly load historical data
                    logging.info("Attempting to explicitly load historical data...")
                    
                    # Force the interpreter to load and process the latest data
                    latest_data = temp_interpreter.get_current_market_data()
                    temp_interpreter.get_current_market_condition(latest_data)
                    
                    # Check means again
                    pc1_mean = getattr(temp_interpreter, 'pc1_mean', 0.0)
                    pc2_mean = getattr(temp_interpreter, 'pc2_mean', 0.0)
                    pc3_mean = getattr(temp_interpreter, 'pc3_mean', 0.0)
                    
                    logging.info(f"After forced data load - PC Means: PC1={pc1_mean:.2f}, PC2={pc2_mean:.2f}, PC3={pc3_mean:.2f}")
                
                # Use this properly initialized interpreter
                interpreter = temp_interpreter
            except Exception as e:
                logging.error(f"Error in forced initialization: {str(e)}")
                # Fall back to regular initialization if forced approach fails
                interpreter = MarketConditionInterpreter()
            
            logging.info("MarketConditionInterpreter initialized successfully")
        
        # Initialize BitGet client
        if bitget_client is None:
            logging.info("Initializing BitGet client...")
            # Load configuration for BitGet
            config_path = base_dir / 'config' / 'config.json'
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    api_setup = config.get('bitget', {})
                    if api_setup:
                        bitget_client = BitgetFutures(api_setup)
                        logging.info("BitGet client initialized successfully")
                    else:
                        bitget_client = BitgetFutures()  # Use default without API keys
                        logging.info("BitGet client initialized without API keys")
            else:
                bitget_client = BitgetFutures()  # Use default without API keys
                logging.info("BitGet client initialized without API keys (config not found)")
        
        return True
    except Exception as e:
        logging.error(f"Error initializing clients: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def get_current_price(symbol='ETH/USDT:USDT'):
    """Get current price for a symbol using BitGet client"""
    global bitget_client
    
    try:
        if bitget_client is None:
            initialize_clients()
        
        if bitget_client:
            # Try to get current price from ticker
            try:
                ticker = bitget_client.fetch_ticker(symbol)
                if ticker and 'last' in ticker:
                    return ticker['last']
            except Exception as ticker_error:
                logging.warning(f"Error fetching ticker: {ticker_error}, trying OHLCV")
            
            # If ticker fails, try OHLCV
            try:
                recent_candles = bitget_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe='1h',
                    limit=1
                )
                if recent_candles and len(recent_candles) > 0:
                    return recent_candles[0][4]  # Close price
            except Exception as ohlcv_error:
                logging.error(f"Error fetching OHLCV: {ohlcv_error}")
    except Exception as e:
        logging.error(f"Error getting current price: {str(e)}")
    
    return None

def initialize_interpreter():
    """Initialize the market interpreter if not already done"""
    global interpreter
    try:
        if interpreter is None:
            logging.info("Initializing MarketConditionInterpreter...")
            interpreter = MarketConditionInterpreter()
            logging.info("MarketConditionInterpreter initialized successfully")
            return True
        return True
    except Exception as e:
        logging.error(f"Error initializing interpreter: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def update_market_data():
    """Update market data using the interpreter"""
    global market_data, interpreter, bitget_client
    
    try:
        # Initialize interpreter and BitGet client if needed
        if interpreter is None or bitget_client is None:
            success = initialize_clients()
            if not success:
                logging.error("Failed to initialize clients")
                return False
        
        # Get current market condition
        result = interpreter.get_current_market_condition()
        
        # Check if we have a valid result
        if 'error' in result:
            logging.error(f"Error getting market data: {result['error']}")
            return False
        
        # Get current price directly using BitGet client
        current_price = get_current_price()
        if current_price:
            result['price'] = current_price
            logging.info(f"Current ETH/USDT price: {current_price}")
        
        # Try to get the raw features for additional indicators
        try:
            # Get current market data with raw features
            data = interpreter.get_current_market_data()
            
            # Extract the last row for current values
            if data is not None and not data.empty:
                last_row = data.iloc[-1]
                
                # Add technical indicators
                if 'rsi' in last_row:
                    result['rsi'] = last_row['rsi']
                if 'adx' in last_row:
                    result['adx'] = last_row['adx']
                if 'macd_diff' in last_row:
                    result['macd_diff'] = last_row['macd_diff']
                if 'bb_width' in last_row:
                    result['bb_width'] = last_row['bb_width']
        except Exception as e:
            logging.warning(f"Could not get additional indicators: {str(e)}")
        
        # Add debug logging for PC values and classification
        try:
            # Log the PC values
            pc1 = result.get('pc1', 0.0)
            pc2 = result.get('pc2', 0.0)
            pc3 = result.get('pc3', 0.0)
            
            # Log the PCA means from the interpreter
            pc1_mean = getattr(interpreter, 'pc1_mean', 0.0)
            pc2_mean = getattr(interpreter, 'pc2_mean', 0.0)
            pc3_mean = getattr(interpreter, 'pc3_mean', 0.0)
            
            logging.info(f"PC Values - PC1: {pc1:.2f}, PC2: {pc2:.2f}, PC3: {pc3:.2f}")
            logging.info(f"PC Means  - PC1: {pc1_mean:.2f}, PC2: {pc2_mean:.2f}, PC3: {pc3_mean:.2f}")
            
            # If means are all close to zero, this is likely incorrect
            if abs(pc1_mean) < 0.01 and abs(pc2_mean) < 0.01 and abs(pc3_mean) < 0.01:
                logging.warning("All PC means are close to zero, which is likely incorrect.")
                logging.warning("The historical data may not be properly initialized.")
                logging.warning("Try to restart the app or run market_interpreter.py directly once.")
                
                # Force the actual market regime calculation using values we calculate here
                # This ensures we match the logic of market_interpreter.py
                if pc1 > 0 and pc2 > 0:
                    corrected_regime = "Strong Trending Market"
                elif pc1 > 0 and pc2 < 0:
                    corrected_regime = "Momentum Without Trend"
                elif pc1 < 0 and pc2 < 0:
                    corrected_regime = "Choppy/Noisy Market"
                else:
                    corrected_regime = "Undefined"
                
                # Log the corrected regime
                if result['market_regime'] != corrected_regime:
                    logging.warning(f"Overriding incorrect market regime '{result['market_regime']}' with '{corrected_regime}'")
                    result['market_regime'] = corrected_regime
            
            # Log the classification condition
            if pc1 > pc1_mean and pc2 > pc2_mean:
                expected_regime = "Strong Trending Market"
            elif pc1 > pc1_mean and pc2 < pc2_mean:
                expected_regime = "Momentum Without Trend"
            elif pc1 < pc1_mean and pc2 < pc2_mean:
                expected_regime = "Choppy/Noisy Market"
            else:
                expected_regime = "Undefined"
                
            logging.info(f"Market Regime - Actual: {result['market_regime']}, Expected: {expected_regime}")
            
            # If they don't match, add a detailed explanation
            if result['market_regime'] != expected_regime:
                logging.warning(f"Market regime mismatch! Conditions:")
                logging.warning(f"PC1 ({pc1:.2f}) {'>' if pc1 > pc1_mean else '<='} Mean ({pc1_mean:.2f})")
                logging.warning(f"PC2 ({pc2:.2f}) {'>' if pc2 > pc2_mean else '<='} Mean ({pc2_mean:.2f})")
            
            # Add direct check of interpreter's classify_market_regime method
            direct_classification = interpreter.classify_market_regime(pc1, pc2)
            logging.info(f"Direct classification result: {direct_classification}")
            
        except Exception as e:
            logging.error(f"Error in debug logging: {str(e)}")
            
        # Convert NumPy values to Python native types
        result = convert_numpy_values(result)
        
        # Add timestamp for the web display
        result['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure all keys are present with default values if needed
        default_keys = {'market_regime', 'trade_signal', 'pc1', 'pc2', 'pc3'}
        for key in default_keys:
            if key not in result:
                if key in ['pc1', 'pc2', 'pc3']:
                    result[key] = 0.0
                else:
                    result[key] = 'Unknown'
        
        # Store the current data in history (limit to 24 entries)
        history_entry = result.copy()
        market_data['history'].insert(0, history_entry)
        market_data['history'] = market_data['history'][:24]  # Keep only the last 24 updates
        
        # Update the current market data
        market_data.update(result)
        
        logging.info(f"Market data updated: {result['market_regime']} - {result['trade_signal']}")
        return True
        
    except Exception as e:
        logging.error(f"Error updating market data: {str(e)}")
        logging.error(traceback.format_exc())
        return False

# Add a route to run the market_interpreter.py directly for comparison
@app.route('/run-directly')
def run_directly():
    """Run market_interpreter.py directly for comparison"""
    try:
        # Create a new instance to ensure we're not using cached data
        direct_interpreter = MarketConditionInterpreter()
        
        # Get current market condition
        result = direct_interpreter.get_current_market_condition()
        
        # Get the PC values and means for comparison
        pc1 = result.get('pc1', 0.0)
        pc2 = result.get('pc2', 0.0)
        pc3 = result.get('pc3', 0.0)
        
        pc1_mean = getattr(direct_interpreter, 'pc1_mean', 0.0)
        pc2_mean = getattr(direct_interpreter, 'pc2_mean', 0.0)
        
        # Log the values
        logging.info(f"DIRECT RUN - PC Values: PC1={pc1:.2f}, PC2={pc2:.2f}, PC3={pc3:.2f}")
        logging.info(f"DIRECT RUN - PC Means: PC1={pc1_mean:.2f}, PC2={pc2_mean:.2f}")
        logging.info(f"DIRECT RUN - Market Regime: {result['market_regime']}")
        logging.info(f"DIRECT RUN - Trade Signal: {result['trade_signal']}")
        
        # Return the results as JSON
        return jsonify({
            'market_regime': result['market_regime'],
            'trade_signal': result['trade_signal'],
            'pc1': float(pc1),
            'pc2': float(pc2),
            'pc3': float(pc3),
            'pc1_mean': float(pc1_mean),
            'pc2_mean': float(pc2_mean)
        })
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error running directly: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

# Add a route to force initialization of the interpreter
@app.route('/force-initialize')
def force_initialize():
    """Force reinitialization of the interpreter"""
    global interpreter
    try:
        # Set to None to force reinitialization in the next update
        interpreter = None
        success = initialize_clients()
        
        if success:
            # Get the means as verification
            pc1_mean = getattr(interpreter, 'pc1_mean', 0.0)
            pc2_mean = getattr(interpreter, 'pc2_mean', 0.0)
            pc3_mean = getattr(interpreter, 'pc3_mean', 0.0)
            
            # Trigger an update
            update_market_data()
            
            return jsonify({
                "status": "success", 
                "message": "Interpreter reinitialized successfully",
                "pc1_mean": float(pc1_mean),
                "pc2_mean": float(pc2_mean),
                "pc3_mean": float(pc3_mean)
            })
        else:
            return jsonify({"status": "error", "message": "Failed to reinitialize interpreter"})
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error in force initialize: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

# Add a route to analyze PC3 relationship with indicators
@app.route('/analyze-pc3')
def analyze_pc3():
    """Analyze the relationship between PC3 and raw indicators"""
    try:
        if interpreter is None:
            initialize_clients()
        
        # Get current market data with raw features
        data = interpreter.get_current_market_data()
        
        # Extract the last row for current values
        last_row = data.iloc[-1].copy() if data is not None and not data.empty else None
        
        if last_row is None:
            return jsonify({"status": "error", "message": "No market data available"})
        
        # Get transformed data for PC values
        transformed_data = interpreter.transform_data(data.iloc[-1:])
        pc3 = transformed_data['PC3'].iloc[0] if 'PC3' in transformed_data.columns else 0.0
        
        # Get the third PCA component loadings
        pc3_loadings = None
        if hasattr(interpreter, 'pca_components') and len(interpreter.pca_components) >= 3:
            pc3_loadings = dict(zip(interpreter.feature_names, interpreter.pca_components[2]))
            
            # Sort to get the most influential features for PC3
            sorted_loadings = sorted(pc3_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pc3_features = sorted_loadings[:5]  # Get top 5 most influential features
        else:
            top_pc3_features = []
        
        # Get specific indicators of interest
        rsi = float(last_row['rsi']) if 'rsi' in last_row else None
        macd = float(last_row['macd_diff']) if 'macd_diff' in last_row else None
        adx = float(last_row['adx']) if 'adx' in last_row else None
        
        # Convert all of last_row to Python native types for display
        last_row_dict = {k: float(v) if isinstance(v, np.number) else v for k, v in last_row.items()}
        
        # Create result with analysis
        result = {
            "pc3_value": float(pc3),
            "rsi": rsi,
            "macd": macd,
            "adx": adx,
            "top_pc3_features": [{
                "feature": feature, 
                "loading": float(loading),
                "value": float(last_row.get(feature, 0)) if feature in last_row else None
            } for feature, loading in top_pc3_features],
            "interpretation": {
                "expected": "Positive PC3 = High MACD, Low RSI; Negative PC3 = Low MACD, High RSI",
                "actual": "The actual relationship in your model might be different. Check the loading weights."
            },
            "all_features": last_row_dict
        }
        
        # Log the analysis for debugging
        logging.info(f"PC3 Analysis - PC3: {pc3:.4f}, RSI: {rsi:.2f}, MACD: {macd:.4f}")
        if top_pc3_features:
            logging.info(f"Top PC3 Features: {top_pc3_features}")
        
        return jsonify(result)
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error analyzing PC3: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

# Add a route to analyze PC1 relationship with indicators
@app.route('/analyze-pc1')
def analyze_pc1():
    """Analyze the relationship between PC1 and raw indicators"""
    try:
        if interpreter is None:
            initialize_clients()
        
        # Get current market data with raw features
        data = interpreter.get_current_market_data()
        
        # Extract the last row for current values
        last_row = data.iloc[-1].copy() if data is not None and not data.empty else None
        
        if last_row is None:
            return jsonify({"status": "error", "message": "No market data available"})
        
        # Get transformed data for PC values
        transformed_data = interpreter.transform_data(data.iloc[-1:])
        pc1 = transformed_data['PC1'].iloc[0] if 'PC1' in transformed_data.columns else 0.0
        
        # Get the first PCA component loadings
        pc1_loadings = None
        if hasattr(interpreter, 'pca_components') and len(interpreter.pca_components) >= 1:
            pc1_loadings = dict(zip(interpreter.feature_names, interpreter.pca_components[0]))
            
            # Sort to get the most influential features for PC1
            sorted_loadings = sorted(pc1_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pc1_features = sorted_loadings[:5]  # Get top 5 most influential features
        else:
            top_pc1_features = []
        
        # Get specific indicators of interest
        rsi = float(last_row['rsi']) if 'rsi' in last_row else None
        macd = float(last_row['macd_diff']) if 'macd_diff' in last_row else None
        adx = float(last_row['adx']) if 'adx' in last_row else None
        
        # Convert all of last_row to Python native types for display
        last_row_dict = {k: float(v) if isinstance(v, np.number) else v for k, v in last_row.items()}
        
        # Create result with analysis
        result = {
            "pc1_value": float(pc1),
            "rsi": rsi,
            "macd": macd,
            "adx": adx,
            "top_pc1_features": [{
                "feature": feature, 
                "loading": float(loading),
                "value": float(last_row.get(feature, 0)) if feature in last_row else None
            } for feature, loading in top_pc1_features],
            "interpretation": {
                "expected": "Positive PC1 = Strong momentum (high MACD); Negative PC1 = Weak momentum (low MACD)",
                "actual": "Check the feature loadings to see what PC1 actually represents in your model."
            },
            "all_features": last_row_dict
        }
        
        # Log the analysis for debugging
        logging.info(f"PC1 Analysis - PC1: {pc1:.4f}, RSI: {rsi:.2f}, MACD: {macd:.4f}")
        if top_pc1_features:
            logging.info(f"Top PC1 Features: {top_pc1_features}")
        
        return jsonify(result)
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error analyzing PC1: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

# Add a route to analyze PC2 relationship with indicators
@app.route('/analyze-pc2')
def analyze_pc2():
    """Analyze the relationship between PC2 and raw indicators"""
    try:
        if interpreter is None:
            initialize_clients()
        
        # Get current market data with raw features
        data = interpreter.get_current_market_data()
        
        # Extract the last row for current values
        last_row = data.iloc[-1].copy() if data is not None and not data.empty else None
        
        if last_row is None:
            return jsonify({"status": "error", "message": "No market data available"})
        
        # Get transformed data for PC values
        transformed_data = interpreter.transform_data(data.iloc[-1:])
        pc2 = transformed_data['PC2'].iloc[0] if 'PC2' in transformed_data.columns else 0.0
        
        # Get the second PCA component loadings
        pc2_loadings = None
        if hasattr(interpreter, 'pca_components') and len(interpreter.pca_components) >= 2:
            pc2_loadings = dict(zip(interpreter.feature_names, interpreter.pca_components[1]))
            
            # Sort to get the most influential features for PC2
            sorted_loadings = sorted(pc2_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pc2_features = sorted_loadings[:5]  # Get top 5 most influential features
        else:
            top_pc2_features = []
        
        # Get specific indicators of interest
        rsi = float(last_row['rsi']) if 'rsi' in last_row else None
        macd = float(last_row['macd_diff']) if 'macd_diff' in last_row else None
        adx = float(last_row['adx']) if 'adx' in last_row else None
        bb_width = float(last_row['bb_width']) if 'bb_width' in last_row else None
        
        # Convert all of last_row to Python native types for display
        last_row_dict = {k: float(v) if isinstance(v, np.number) else v for k, v in last_row.items()}
        
        # Create result with analysis
        result = {
            "pc2_value": float(pc2),
            "rsi": rsi,
            "macd": macd,
            "adx": adx,
            "bb_width": bb_width,
            "top_pc2_features": [{
                "feature": feature, 
                "loading": float(loading),
                "value": float(last_row.get(feature, 0)) if feature in last_row else None
            } for feature, loading in top_pc2_features],
            "interpretation": {
                "expected": "Positive PC2 = High volatility/RSI, trending market; Negative PC2 = Low volatility/RSI, choppy market",
                "actual": "Check the feature loadings to see what PC2 actually represents in your model."
            },
            "all_features": last_row_dict
        }
        
        # Log the analysis for debugging
        logging.info(f"PC2 Analysis - PC2: {pc2:.4f}, RSI: {rsi:.2f}, ADX: {adx:.2f}")
        if top_pc2_features:
            logging.info(f"Top PC2 Features: {top_pc2_features}")
        
        return jsonify(result)
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error analyzing PC2: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

# Context processor to add current date to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Routes
@app.route('/')
def index():
    """Main page displaying current market conditions"""
    return render_template('index.html', market_data=market_data)

@app.route('/history')
def history():
    """Page displaying historical market conditions"""
    return render_template('history.html', market_data=market_data)

@app.route('/api/market-data')
def api_market_data():
    """API endpoint for getting current market data in JSON format"""
    return jsonify({
        'market_regime': market_data['market_regime'],
        'trade_signal': market_data['trade_signal'],
        'pc1': market_data['pc1'],
        'pc2': market_data['pc2'],
        'pc3': market_data['pc3'],
        'price': market_data.get('price', 0.0),
        'rsi': market_data.get('rsi', 0.0),
        'adx': market_data.get('adx', 0.0),
        'macd_diff': market_data.get('macd_diff', 0.0),
        'bb_width': market_data.get('bb_width', 0.0),
        'last_updated': market_data['last_updated']
    })

@app.route('/update', methods=['GET'])
def manual_update():
    """Endpoint to manually trigger a data update"""
    try:
        success = update_market_data()
        if success:
            return jsonify({"status": "success", "message": "Market data updated successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to update market data"})
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error in manual update: {error_message}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_message})

def start_scheduler():
    """Start the background scheduler for periodic updates"""
    try:
        scheduler = BackgroundScheduler()
        # Update every hour
        scheduler.add_job(update_market_data, 'interval', hours=1)
        # Also update on startup
        scheduler.add_job(update_market_data, 'date')
        scheduler.start()
        logging.info("Scheduler started")
        return scheduler
    except Exception as e:
        logging.error(f"Error starting scheduler: {str(e)}")
        logging.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    # Initialize the interpreter and BitGet client
    initialize_clients()
    
    # Start the scheduler
    scheduler = start_scheduler()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 