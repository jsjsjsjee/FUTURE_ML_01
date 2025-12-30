import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data():
    """Load and preprocess the Superstore data"""
    # Load with correct encoding
    df = pd.read_csv("data/Superstore.csv", encoding='latin1')
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Convert date columns - handle multiple date formats
    df = convert_dates(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    return df

def convert_dates(df):
    """Convert date columns with flexible format handling"""
    
    # List of possible date columns
    date_columns = ['Order Date', 'Ship Date']
    
    for col in date_columns:
        if col in df.columns:
            # First, try to infer the format
            try:
                # Try parsing with dayfirst=False (US format: MM/DD/YYYY)
                df[col] = pd.to_datetime(df[col], dayfirst=False)
                print(f"Successfully parsed {col} with US format (MM/DD/YYYY)")
            except:
                try:
                    # Try parsing with dayfirst=True (European format: DD/MM/YYYY)
                    df[col] = pd.to_datetime(df[col], dayfirst=True)
                    print(f"Successfully parsed {col} with European format (DD/MM/YYYY)")
                except:
                    try:
                        # Try parsing with format='mixed'
                        df[col] = pd.to_datetime(df[col], format='mixed')
                        print(f"Successfully parsed {col} with mixed format")
                    except Exception as e:
                        print(f"Warning: Could not parse {col}: {str(e)}")
                        # Keep as string if parsing fails
                        
    return df

def engineer_features(df):
    """Engineer new features for better forecasting"""
    
    # Ensure date columns are datetime
    for col in ['Order Date', 'Ship Date']:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Basic date features
    if 'Order Date' in df.columns:
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Quarter'] = df['Order Date'].dt.quarter
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['DayOfMonth'] = df['Order Date'].dt.day
        df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week
        
        # Create month-year period for grouping
        df['MonthPeriod'] = df['Order Date'].dt.to_period('M')
    
    # Holiday indicators (US holidays - adjust as needed)
    df['IsHolidayMonth'] = df['Month'].isin([11, 12]) if 'Month' in df.columns else False
    df['IsYearEnd'] = df['Month'].isin([12, 1]) if 'Month' in df.columns else False
    
    # Season indicators
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season) if 'Month' in df.columns else 'Unknown'
    
    # Business metrics
    if 'Profit' in df.columns and 'Sales' in df.columns:
        df['ProfitMargin'] = (df['Profit'] / df['Sales']).replace([np.inf, -np.inf], 0).fillna(0)
        df['DiscountPct'] = df['Discount'] * 100 if 'Discount' in df.columns else 0
    
    # Shipping time
    if 'Ship Date' in df.columns and 'Order Date' in df.columns:
        df['ShippingDays'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Price tier
    if 'Sales' in df.columns:
        df['PriceTier'] = pd.cut(df['Sales'], 
                                bins=[0, 50, 200, 500, float('inf')],
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Customer frequency (simplified)
    if 'Customer ID' in df.columns:
        customer_orders = df.groupby('Customer ID').size()
        df['CustomerOrderCount'] = df['Customer ID'].map(customer_orders)
    
    return df

def get_filter_options(df):
    """Get unique values for filters"""
    options = {}
    
    if 'Region' in df.columns:
        options['regions'] = sorted(df['Region'].unique().tolist())
    
    if 'Category' in df.columns:
        options['categories'] = sorted(df['Category'].unique().tolist())
    
    if 'Year' in df.columns:
        options['years'] = sorted(df['Year'].unique().tolist())
    
    if 'Segment' in df.columns:
        options['segments'] = sorted(df['Segment'].unique().tolist())
    
    return options