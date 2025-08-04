import numpy as np
import pandas as pd
import re

def analyze_csv(df, task):
    """Analyze CSV data based on task"""
    if "count" in task.lower():
        return [len(df)]
    return ["No analysis implemented for this task."]

def analyze_scraped_data(df, task):
    """Analyze scraped Wikipedia data"""
    answers = []

    # 1. How many $2bn movies before 2020?
    if "2 bn" in task or "$2 bn" in task or "2bn" in task:
        count_2bn = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2020)].shape[0]
        answers.append(count_2bn)
    else:
        answers.append(None)

    # 2. Earliest $1.5bn+ film?
    if "1.5 bn" in task or "$1.5 bn" in task or "1.5bn" in task:
        df_15 = df[df['Worldwide gross'] >= 1_500_000_000]
        if not df_15.empty:
            earliest = df_15.sort_values('Year').iloc[0]['Title']
            answers.append(str(earliest))
        else:
            answers.append(None)
    else:
        answers.append(None)

    # 3. Correlation between Rank and Peak?
    if "correlation" in task.lower():
        if 'Rank' in df.columns and 'Peak' in df.columns:
            corr = np.corrcoef(df['Rank'], df['Peak'])[0,1]
            answers.append(float(corr))
        else:
            answers.append(None)
    else:
        answers.append(None)

    # 4. Top 5 films by worldwide gross
    if "top 5" in task.lower() and "worldwide gross" in task.lower():
        top5 = df.sort_values('Worldwide gross', ascending=False).head(5)['Title'].tolist()
        answers.append(top5)
    else:
        answers.append(None)

    # 5. Average worldwide gross for films after 2010
    if "average" in task.lower() and "after 2010" in task.lower():
        avg = df[df['Year'] > 2010]['Worldwide gross'].mean()
        answers.append(float(avg) if not np.isnan(avg) else None)
    else:
        answers.append(None)

    # 6. Film with highest Peak rank
    if "highest peak" in task.lower():
        idx = df['Peak'].idxmax()
        film = df.loc[idx, 'Title'] if not np.isnan(idx) else None
        answers.append(film)
    else:
        answers.append(None)

    # 7. Median worldwide gross
    if "median" in task.lower() and "worldwide gross" in task.lower():
        median = df['Worldwide gross'].median()
        answers.append(float(median) if not np.isnan(median) else None)
    else:
        answers.append(None)

    # 8. Films released in 2019
    if "released in 2019" in task.lower():
        films_2019 = df[df['Year'] == 2019]['Title'].tolist()
        answers.append(films_2019)
    else:
        answers.append(None)

    # 9. Year with most $1bn+ films
    if "most" in task.lower() and "1 bn" in task.lower() and "year" in task.lower():
        df_1bn = df[df['Worldwide gross'] >= 1_000_000_000]
        if not df_1bn.empty:
            year = df_1bn['Year'].value_counts().idxmax()
            answers.append(int(year))
        else:
            answers.append(None)
    else:
        answers.append(None)

    return answers

def analyze_high_court_data(df, task):
    """Analyze Indian High Court data"""
    answers = {}
    
    # Which high court disposed the most cases from 2019-2022?
    if "high court" in task.lower() and "most cases" in task.lower():
        if not df.empty and 'court' in df.columns and 'case_count' in df.columns:
            most_cases = df.loc[df['case_count'].idxmax(), 'court']
            answers["Which high court disposed the most cases from 2019 - 2022?"] = most_cases
        else:
            answers["Which high court disposed the most cases from 2019 - 2022?"] = "No data available"
    
    # Regression slope calculation
    if "regression slope" in task.lower():
        if not df.empty and 'year' in df.columns and 'avg_delay' in df.columns:
            x = df['year'].values
            y = df['avg_delay'].values
            if len(x) > 1:
                slope = (len(x) * sum(x*y) - sum(x) * sum(y)) / (len(x) * sum(x*x) - sum(x)**2)
                answers["What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"] = float(slope)
            else:
                answers["What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"] = 0.0
        else:
            answers["What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"] = 0.0
    
    return answers

def extract_numeric_value(text):
    """Extract numeric value from text"""
    if isinstance(text, (int, float)):
        return text
    if isinstance(text, str):
        # Remove common currency symbols and commas
        cleaned = re.sub(r'[\$,]', '', text)
        # Extract first number
        match = re.search(r'(\d+\.?\d*)', cleaned)
        if match:
            return float(match.group(1))
    return None

def calculate_correlation(df, col1, col2):
    """Calculate correlation between two columns"""
    if col1 in df.columns and col2 in df.columns:
        return df[col1].corr(df[col2])
    return None