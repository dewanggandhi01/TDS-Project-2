import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd

def create_plot(df, task):
    """Create scatterplot of Rank vs Peak with dotted red regression line"""
    try:
        if 'Rank' in df.columns and 'Peak' in df.columns:
            # Remove NaN values
            valid_data = df.dropna(subset=['Rank', 'Peak'])
            
            if len(valid_data) == 0:
                return "No valid data for plotting."
            
            x = valid_data['Rank']
            y = valid_data['Peak']
            
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, alpha=0.6, s=30)
            
            # Regression line
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x + b, 'r:', linewidth=2, label=f'Regression line (slope: {m:.3f})')
            
            plt.xlabel('Rank', fontsize=12)
            plt.ylabel('Peak', fontsize=12)
            plt.title('Rank vs Peak Scatterplot with Regression Line', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"
        else:
            return "Required columns (Rank, Peak) not found in data."
    except Exception as e:
        print(f"Error creating plot: {e}")
        return f"Error creating plot: {str(e)}"

def create_high_court_plot():
    """Create plot for high court data analysis"""
    try:
        # Create sample data for demonstration
        years = [2019, 2020, 2021, 2022, 2023]
        delays = [30, 35, 40, 45, 50]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(years, delays, s=100, alpha=0.7)
        
        # Regression line
        m, b = np.polyfit(years, delays, 1)
        plt.plot(years, m*np.array(years) + b, 'r:', linewidth=2, label=f'Regression line (slope: {m:.2f})')
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Delay (days)', fontsize=12)
        plt.title('Year vs Average Delay in Court Cases', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error creating high court plot: {e}")
        return f"Error creating plot: {str(e)}"

def create_custom_plot(df, x_col, y_col, title="Custom Plot", x_label=None, y_label=None):
    """Create a custom scatterplot with regression line"""
    try:
        if x_col not in df.columns or y_col not in df.columns:
            return f"Required columns ({x_col}, {y_col}) not found in data."
        
        # Remove NaN values
        valid_data = df.dropna(subset=[x_col, y_col])
        
        if len(valid_data) == 0:
            return "No valid data for plotting."
        
        x = valid_data[x_col]
        y = valid_data[y_col]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6, s=30)
        
        # Regression line
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m*x + b, 'r:', linewidth=2, label=f'Regression line (slope: {m:.3f})')
        
        plt.xlabel(x_label or x_col, fontsize=12)
        plt.ylabel(y_label or y_col, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error creating custom plot: {e}")
        return f"Error creating plot: {str(e)}"