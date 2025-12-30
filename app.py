from flask import Flask, render_template, jsonify, request, redirect, url_for
from model.forecast import generate_forecast, get_filtered_data
from model.data_processor import load_and_preprocess_data
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
    # Redirect to dashboard page which has all the data
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    try:
        # Load data
        df = load_and_preprocess_data()
        
        # Get base forecast - now returns 3 items
        monthly_sales, forecast_df, insights = generate_forecast()
        
        # Convert to JSON serializable format
        monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%Y-%m')
        forecast_df['Month'] = forecast_df['Month'].dt.strftime('%Y-%m')
        
        # ... rest of your code

        # Get top products
        top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
        
        # Get regional data
        regional_sales = df.groupby('Region')['Sales'].sum().reset_index()
        
        # Get category data
        category_sales = df.groupby('Category')['Sales'].sum().reset_index()
        category_profit = df.groupby('Category')['Profit'].sum().reset_index()
        
        # Get monthly trends
        monthly_trends = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
        monthly_trends['Order Date'] = monthly_trends['Order Date'].dt.strftime('%Y-%m')
        
        return render_template('dashboard.html',
                             monthly_sales=monthly_sales.to_dict('records'),
                             forecast_df=forecast_df.to_dict('records'),
                             insights=insights,
                             top_products=top_products.to_dict('records'),
                             regional_sales=regional_sales.to_dict('records'),
                             category_sales=category_sales.to_dict('records'),
                             category_profit=category_profit.to_dict('records'),
                             monthly_trends=monthly_trends.to_dict('records'))
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/insights')
def insights_page():
    try:
        # Load data for insights page
        df = load_and_preprocess_data()
        
        # Get forecast data
        monthly_sales, forecast_df, insights = generate_forecast()
        
        # Get regional data
        regional_sales = df.groupby('Region')['Sales'].sum().reset_index()
        
        # Format dates for display
        monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%Y-%m')
        forecast_df['Month'] = forecast_df['Month'].dt.strftime('%Y-%m')
        
        return render_template('insights.html',
                             insights=insights,
                             regional_sales=regional_sales.to_dict('records'),
                             monthly_sales=monthly_sales.to_dict('records'),
                             forecast_df=forecast_df.to_dict('records'))
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/about')
def about():
    try:
        # Load data for about page
        df = load_and_preprocess_data()
        monthly_sales, forecast_df, insights = generate_forecast()
        
        return render_template('about.html',
                             insights=insights)
    
    except Exception as e:
        # If data loading fails, still show about page with default values
        default_insights = {
            'model_accuracy': 85.0,
            'forecast_growth': 12.5,
            'total_sales': 1000000,
            'next_peak_forecast': 'October 2024'
        }
        return render_template('about.html',
                             insights=default_insights)

@app.route('/api/filter', methods=['POST'])
def filter_data():
    try:
        data = request.json
        region = data.get('region', 'All')
        category = data.get('category', 'All')
        year = data.get('year', 'All')
        
        filtered_data = get_filtered_data(region, category, year)
        
        return jsonify({
            'monthly_sales': filtered_data[0].to_dict('records'),
            'forecast_df': filtered_data[1].to_dict('records'),
            'insights': filtered_data[2]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights')
def get_insights():
    try:
        df = load_and_preprocess_data()
        
        # Calculate various insights
        insights = {
            'total_sales': float(df['Sales'].sum()),
            'total_profit': float(df['Profit'].sum()),
            'avg_order_value': float(df['Sales'].mean()),
            'total_orders': int(df['Order ID'].nunique()),
            'top_category': df.groupby('Category')['Sales'].sum().idxmax(),
            'most_profitable_region': df.groupby('Region')['Profit'].sum().idxmax(),
            'growth_rate': calculate_growth_rate(df),
            'peak_month': get_peak_month(df),
            'low_season': get_low_season(df)
        }
        
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_growth_rate(df):
    df['Year'] = pd.to_datetime(df['Order Date']).dt.year
    yearly_sales = df.groupby('Year')['Sales'].sum()
    if len(yearly_sales) > 1:
        growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[-2]) / yearly_sales.iloc[-2]) * 100
        return round(growth, 2)
    return 0

def get_peak_month(df):
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
    return monthly_sales.idxmax().strftime('%B %Y')

def get_low_season(df):
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
    return monthly_sales.idxmin().strftime('%B %Y')

if __name__ == '__main__':
    app.run(debug=True, port=5000)