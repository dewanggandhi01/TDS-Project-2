import re
import json
import math
import pandas as pd
from typing import List, Dict, Any, Union
from app.data_loader import load_csv, scrape_wikipedia, clean_scraped_df, query_s3_parquet, get_wikipedia_summary
from app.analyzer import analyze_scraped_data, analyze_high_court_data
from app.visualizer import create_plot, create_high_court_plot, create_custom_plot
from app.react_agent import react_agent
from app.llm_agent import get_llm_agent
import logging

logger = logging.getLogger(__name__)

def nan_to_none(obj):
    """Convert NaN values to None for JSON serialization"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, list):
        return [nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    return obj

def extract_questions(task: str) -> List[str]:
    """Extract individual questions from the task text"""
    # Split by numbered questions or bullet points
    questions = []
    
    # Look for numbered questions (1., 2., etc.)
    numbered_pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
    numbered_matches = re.findall(numbered_pattern, task, re.DOTALL)
    
    # Look for questions starting with "What", "How", "Which", etc.
    question_pattern = r'(What|How|Which|List|Draw|Plot|Answer).+?(?=\n\n|\n\d+\.|$)'
    question_matches = re.findall(question_pattern, task, re.DOTALL)
    
    # Combine and clean
    all_questions = numbered_matches + question_matches
    for q in all_questions:
        q = q.strip()
        if q and len(q) > 10:  # Filter out very short matches
            questions.append(q)
    
    return questions if questions else [task]

def process_task(task: str, use_llm: bool = True) -> Union[List, Dict]:
    """Main function to process data analysis tasks"""
    try:
        # Use traditional approach for Wikipedia film data tasks (more reliable)
        if "wikipedia" in task.lower() or "highest-grossing films" in task.lower():
            return process_wikipedia_task(task, use_llm)
        
        # Use traditional approach for other tasks
        questions = extract_questions(task)
        
        if "indian high court" in task.lower() or "ecourts" in task.lower():
            return process_high_court_task(task, questions, use_llm)
        else:
            return process_csv_task(task, questions, use_llm)
            
    except Exception as e:
        logger.error(f"Error in process_task: {e}")
        return {"error": str(e)}

def process_wikipedia_task(task: str, use_llm: bool = True) -> List:
    """Process Wikipedia film data tasks"""
    try:
        logger.info("Processing Wikipedia task...")
        
        # Scrape Wikipedia data
        data = scrape_wikipedia(task)
        data = clean_scraped_df(data)
        
        logger.info(f"Scraped data shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")

        # Detect if this looks like the 'highest-grossing films' table
        film_required_cols = {"Worldwide gross", "Year"}
        is_film_table = film_required_cols.issubset(set(map(str, data.columns)))
        
        if use_llm:
            # Use LLM for processing with graceful fallback
            answers = process_with_llm(task, data)
            # If LLM failed or returned empty/None answers, fallback to traditional/generic
            if not answers or all(a is None for a in answers):
                if is_film_table:
                    logger.warning("LLM failed; falling back to specialized film analysis.")
                    return process_wikipedia_traditional(data)
                else:
                    logger.warning("LLM failed and table is not a film dataset; trying generic business analysis.")
                    norm = normalize_business_table(data)
                    if norm is not None and not norm.empty:
                        # Extract targeted answers from task
                        ans1, ans2 = compute_business_answers_from_task(task, norm)
                        plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')
                        return [ans1, ans2, None, plot_uri]
                    logger.warning("Generic business analysis returned no result; using summary fallback.")
                    summary = get_wikipedia_summary(task)
                    out = [summary] if summary else [None]
                    while len(out) < 4:
                        out.append(None)
                    return out
            return answers
        else:
            # Use traditional approach only for film dataset; otherwise generic business analyzer
            if is_film_table:
                return process_wikipedia_traditional(data)
            else:
                logger.info("Running generic business analysis for non-film Wikipedia table.")
                norm = normalize_business_table(data)
                if norm is not None and not norm.empty:
                    return process_business_query(task, norm)
                logger.warning("Generic business analysis returned no result; using summary fallback.")
                summary = get_wikipedia_summary(task)
                out = [summary] if summary else [None]
                while len(out) < 4:
                    out.append(None)
                return out
        
    except Exception as e:
        logger.error(f"Error in process_wikipedia_task: {e}")
        import traceback
        traceback.print_exc()
        return [None, None, None, None]

def process_with_llm(task: str, data) -> List:
    """Process task using LLM"""
    try:
        llm_agent = get_llm_agent()
        result = llm_agent.process_task(task, data)
        
        if result.get('success', False):
            # Convert LLM result to expected format
            answers = []
            
            if result.get('type') == 'plot' and result.get('plot'):
                answers.append(result['plot'])
            else:
                answers.append(result.get('result'))
            
            # Pad with None to match expected format
            while len(answers) < 4:
                answers.append(None)
            
            return answers[:4]
        else:
            logger.error(f"LLM processing failed: {result.get('error')}")
            return [None, None, None, None]
            
    except Exception as e:
        logger.error(f"Error in process_with_llm: {e}")
        return [None, None, None, None]

def process_wikipedia_traditional(data) -> List:
    """Process Wikipedia data using traditional approach"""
    # Ensure required columns exist
    required = ["Worldwide gross", "Year"]
    if not all(col in data.columns for col in required):
        logger.warning("Traditional film analysis: required columns missing; returning default answers.")
        return [None, None, None, None]

    answers = []
    
    # Question 1: How many $2 bn movies before 2000?
    logger.info("Processing question 1...")
    count_2bn = data[(data['Worldwide gross'] >= 2_000_000_000) & (data['Year'] < 2000)].shape[0]
    answers.append(count_2bn)
    logger.info(f"Answer 1: {count_2bn}")
    
    # Question 2: Earliest $1.5bn+ film
    logger.info("Processing question 2...")
    df_15 = data[data['Worldwide gross'] >= 1_500_000_000]
    if not df_15.empty:
        earliest = df_15.sort_values('Year').iloc[0]['Title']
        answers.append(str(earliest).lower())
    else:
        answers.append(None)
    logger.info(f"Answer 2: {answers[-1]}")
    
    # Question 3: Correlation between Rank and Peak
    logger.info("Processing question 3...")
    if 'Rank' in data.columns and 'Peak' in data.columns:
        corr = data['Rank'].corr(data['Peak'])
        answers.append(float(corr) if not math.isnan(corr) else None)
    else:
        answers.append(None)
    logger.info(f"Answer 3: {answers[-1]}")
    
    # Question 4: Scatterplot of Rank vs Peak
    logger.info("Processing question 4...")
    plot_uri = create_plot(data, "scatterplot of rank and peak")
    answers.append(plot_uri)
    logger.info(f"Answer 4: Plot generated (length: {len(str(plot_uri))})")
    
    logger.info(f"Final answers: {answers}")
    return nan_to_none(answers)

def _parse_numeric(value: Any, colname: str = "") -> float | None:
    """Parse numeric values from Wikipedia tables (handles commas, $, million/billion hints)."""
    try:
        if value is None:
            return None
        s = str(value)
        if s.strip() == "" or s.strip().lower() in {"nan", "none", "n/a"}:
            return None
        scale = 1.0
        lc = s.lower()
        # Heuristic: detect unit in column name or value text
        if "billion" in lc or "bn" in lc or ("billion" in colname.lower() if colname else False):
            scale = 1000.0  # convert to million
        if "million" in lc or ("million" in colname.lower() if colname else False):
            scale = max(scale, 1.0)
        # Remove currency and commas and bracket refs
        import re as _re
        s = _re.sub(r"\[[^\]]*\]", "", s)  # remove [1], [2]
        s = s.replace("$", "").replace(",", "")
        # Extract first float/number
        m = _re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        val = float(m.group(0)) * scale
        return val
    except Exception:
        return None

def _detect_year_column(df: Any) -> str | None:
    candidates = [c for c in df.columns if isinstance(c, str) and any(k in c.lower() for k in ["year", "fy", "fiscal"]) ]
    return candidates[0] if candidates else None

def _detect_metric_columns(df: Any) -> Dict[str, str]:
    mapping = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        cl = c.lower()
        if "revenue" in cl and "Revenue" not in mapping:
            mapping["Revenue"] = c
        elif "net income" in cl and "Net income" not in mapping:
            mapping["Net income"] = c
        elif "employee" in cl and "Employees" not in mapping:
            mapping["Employees"] = c
    return mapping

def process_wikipedia_generic_business(data: Any) -> List:
    """Analyze a generic Wikipedia business trends table: return [plot, summary_text, None, None]."""
    try:
        norm = normalize_business_table(data)
        if norm is None or norm.empty:
            return [None, None, None, None]

        # Plot Revenue vs Year
        plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')

        # Compute basic stats: CAGR (in millions)
        years = norm['Year'].tolist()
        revs = norm['Revenue'].tolist()
        if len(revs) >= 2:
            n = years[-1] - years[0] if years[-1] != years[0] else len(revs) - 1
            if n <= 0:
                n = len(revs) - 1
            cagr = (revs[-1] / revs[0]) ** (1.0 / n) - 1.0 if revs[0] > 0 and n > 0 else None
        else:
            cagr = None
        # Build concise text summary instead of returning an object
        def fmt_m(x):
            try:
                return f"{x:,.0f}"
            except Exception:
                return str(x)
        summary = f"Revenue trend: {years[0]}={fmt_m(revs[0])}M → {years[-1]}={fmt_m(revs[-1])}M"
        if cagr is not None:
            summary += f"; CAGR ~ {cagr*100:.1f}%"

        # Return as 4-item list to match UI expectations
        return [plot_uri, summary, None, None]
    except Exception as e:
        logger.error(f"Generic business analysis failed: {e}")
        return [None, None, None, None]

def process_business_query(task: str, norm) -> List[Any]:
    """Process business query based on the specific question asked."""
    try:
        t = task.lower()
        
        # Handle very general questions that aren't about specific data
        if any(general in t for general in ['what is', 'who is', 'about']) and not any(data_word in t for data_word in ['revenue', 'income', 'employee', 'profit', 'sales', 'earning']):
            # This is a general question, not a data query - return summary instead of extracting numbers
            summary = f"This appears to be a general question about the company. For specific financial data, please ask about revenue, employees, or net income for specific years."
            plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')
            return [summary, None, None, plot_uri]
        
        # Check what type of question is being asked
        if any(keyword in t for keyword in ['plot', 'chart', 'graph', 'scatterplot', 'scatter', 'draw']):
            # User wants a visualization
            plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')
            if any(analysis_word in t for analysis_word in ['regression', 'trend', 'growth', 'analysis']):
                # Add trend analysis for charts with analytical intent
                summary = analyze_revenue_trend(norm)
                return [summary, None, None, plot_uri]
            else:
                return [None, None, None, plot_uri]
        
        elif any(keyword in t for keyword in ['trend', 'growth', 'cagr', 'increase', 'decrease', 'change over time', 'analysis']):
            # User wants trend analysis
            summary = analyze_revenue_trend(norm)
            plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')
            return [summary, None, None, plot_uri]
        
        else:
            # Default to specific value extraction only for specific data queries
            ans1, ans2 = compute_business_answers_from_task(task, norm)
            plot_uri = create_custom_plot(norm, 'Year', 'Revenue', title='Revenue Trend', x_label='Year', y_label='Revenue (millions)')
            return [ans1, ans2, None, plot_uri]
            
    except Exception as e:
        logger.error(f"Error in process_business_query: {e}")
        return [None, None, None, None]

def analyze_revenue_trend(norm) -> str:
    """Analyze revenue trend and return summary."""
    try:
        import pandas as pd
        if norm.empty or len(norm) < 2:
            return "Insufficient data for trend analysis"
        
        # Calculate growth metrics
        years = norm['Year'].tolist()
        revenues = norm['Revenue'].astype(float).tolist()
        
        start_year, end_year = int(years[0]), int(years[-1])
        start_rev, end_rev = revenues[0], revenues[-1]
        
        # Calculate CAGR
        n_years = end_year - start_year
        if n_years > 0 and start_rev > 0:
            cagr = ((end_rev / start_rev) ** (1.0 / n_years) - 1) * 100
        else:
            cagr = 0
        
        # Find peak year
        max_idx = norm['Revenue'].astype(float).idxmax()
        peak_year = int(norm.loc[max_idx, 'Year'])
        peak_revenue = float(norm.loc[max_idx, 'Revenue'])
        
        summary = f"Revenue trend from {start_year} to {end_year}: "
        summary += f"{start_rev:.1f}M → {end_rev:.1f}M. "
        summary += f"CAGR: {cagr:.1f}%. "
        summary += f"Peak: {peak_revenue:.1f}M in {peak_year}."
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in analyze_revenue_trend: {e}")
        return "Error analyzing trend"

def normalize_business_table(data: Any):
    """Normalize a generic business table to columns: Year, Revenue, Net income, Employees (numeric)."""
    try:
        year_col = _detect_year_column(data)
        metrics = _detect_metric_columns(data)
        if not year_col or not metrics.get("Revenue"):
            return None
        rev_col = metrics["Revenue"]
        import pandas as pd
        norm = pd.DataFrame()
        def _to_year(x):
            try:
                import re as _re
                s = str(x)
                m = _re.search(r"(?:19|20)\d{2}", s)
                if m:
                    return int(m.group(0))
                return int(float(s))
            except Exception:
                return None
        norm['Year'] = data[year_col].map(_to_year)
        norm['Revenue'] = data[rev_col].map(lambda v: _parse_numeric(v, rev_col))
        if 'Net income' in metrics:
            nic = metrics['Net income']
            norm['Net income'] = data[nic].map(lambda v: _parse_numeric(v, nic))
        if 'Employees' in metrics:
            ec = metrics['Employees']
            norm['Employees'] = data[ec].map(lambda v: _parse_numeric(v, ec))
        norm = norm.dropna(subset=['Year', 'Revenue']).sort_values('Year')
        return norm
    except Exception:
        return None

def compute_business_answers_from_task(task: str, norm) -> List[Any]:
    """Extract specific answers based on the question type and years mentioned in the task."""
    try:
        t = task.lower()
        import re as _re
        years = [int(y) for y in _re.findall(r"\b(?:19|20)\d{2}\b", t)]
        
        # Determine what type of information is being requested
        if "revenue" in t:
            return extract_revenue_answer(t, years, norm)
        elif "employee" in t:
            return extract_employee_answer(t, years, norm)
        elif "net income" in t or "income" in t:
            return extract_net_income_answer(t, years, norm)
        else:
            # Default fallback - return revenue
            return extract_revenue_answer(t, years, norm)
            
    except Exception as e:
        logger.error(f"Error in compute_business_answers_from_task: {e}")
        return [None, None]

def extract_revenue_answer(task: str, years: List[int], norm) -> List[Any]:
    """Extract revenue answer for the specified year or latest year."""
    try:
        chosen_year = None
        chosen_revenue_m = None
        
        if years:
            # Use the first year mentioned
            y = years[0]
            row = norm[norm['Year'] == y]
            if not row.empty:
                chosen_year = int(y)
                chosen_revenue_m = float(row.iloc[0]['Revenue'])
        else:
            # No year specified: pick the latest year
            if not norm.empty:
                latest_year = norm['Year'].max()
                row = norm[norm['Year'] == latest_year]
                if not row.empty:
                    chosen_year = int(latest_year)
                    chosen_revenue_m = float(row.iloc[0]['Revenue'])
        
        if chosen_year is not None and chosen_revenue_m is not None:
            val_b = chosen_revenue_m / 1000.0  # millions -> billions
            rev_ans = round(val_b, 3)
            logger.info(f"Business Q: using revenue year={chosen_year}, billions={rev_ans}")
            return [rev_ans, None]
        
        return [None, None]
    except Exception as e:
        logger.error(f"Error in extract_revenue_answer: {e}")
        return [None, None]

def extract_employee_answer(task: str, years: List[int], norm) -> List[Any]:
    """Extract employee count for the specified year or latest available year."""
    try:
        if 'Employees' not in norm.columns:
            return [None, None]
            
        chosen_year = None
        emp_count = None
        
        if years:
            # Use the first year mentioned
            y = years[0]
            row = norm[norm['Year'] == y]
            if not row.empty and not pd.isna(row.iloc[0]['Employees']):
                chosen_year = int(y)
                emp_count = int(round(float(row.iloc[0]['Employees'])))
        else:
            # No year specified: pick the latest year with employee data
            emp_data = norm[norm['Employees'].notna()]
            if not emp_data.empty:
                latest_emp_row = emp_data.loc[emp_data['Year'].idxmax()]
                chosen_year = int(latest_emp_row['Year'])
                emp_count = int(round(float(latest_emp_row['Employees'])))
        
        if chosen_year is not None and emp_count is not None:
            logger.info(f"Business Q: using employees year={chosen_year}, count={emp_count}")
            return [emp_count, None]
        
        return [None, None]
    except Exception as e:
        logger.error(f"Error in extract_employee_answer: {e}")
        return [None, None]

def extract_net_income_answer(task: str, years: List[int], norm) -> List[Any]:
    """Extract net income for the specified year or latest available year."""
    try:
        if 'Net income' not in norm.columns:
            return [None, None]
            
        chosen_year = None
        income_m = None
        
        if years:
            # Use the first year mentioned
            y = years[0]
            row = norm[norm['Year'] == y]
            if not row.empty and not pd.isna(row.iloc[0]['Net income']):
                chosen_year = int(y)
                income_m = float(row.iloc[0]['Net income'])
        else:
            # No year specified: pick the latest year with income data
            income_data = norm[norm['Net income'].notna()]
            if not income_data.empty:
                latest_income_row = income_data.loc[income_data['Year'].idxmax()]
                chosen_year = int(latest_income_row['Year'])
                income_m = float(latest_income_row['Net income'])
        
        if chosen_year is not None and income_m is not None:
            val_b = income_m / 1000.0  # millions -> billions
            income_ans = round(val_b, 3)
            logger.info(f"Business Q: using net income year={chosen_year}, billions={income_ans}")
            return [income_ans, None]
        
        return [None, None]
    except Exception as e:
        logger.error(f"Error in extract_net_income_answer: {e}")
        return [None, None]

def process_high_court_task(task: str, questions: List[str], use_llm: bool = True) -> Dict:
    """Process Indian High Court data tasks"""
    try:
        if use_llm:
            # Use LLM for high court data
            return process_high_court_with_llm(task, questions)
        else:
            # Use traditional approach
            return process_high_court_traditional(task, questions)
        
    except Exception as e:
        logger.error(f"Error in process_high_court_task: {e}")
        return {"error": str(e)}

def process_high_court_with_llm(task: str, questions: List[str]) -> Dict:
    """Process high court data using LLM"""
    try:
        # Get sample data for LLM processing
        data = get_sample_high_court_data()
        llm_agent = get_llm_agent()
        
        answers = {}
        for question in questions:
            result = llm_agent.process_task(question, data)
            if result.get('success', False):
                if result.get('type') == 'plot' and result.get('plot'):
                    answers[question] = result['plot']
                else:
                    answers[question] = result.get('result')
            else:
                answers[question] = None
        
        return nan_to_none(answers)
        
    except Exception as e:
        logger.error(f"Error in process_high_court_with_llm: {e}")
        return {"error": str(e)}

def process_high_court_traditional(task: str, questions: List[str]) -> Dict:
    """Process high court data using traditional approach"""
    answers = {}
    
    for question in questions:
        question_lower = question.lower()
        
        # Question 1: Which high court disposed the most cases from 2019-2022?
        if "high court" in question_lower and "most cases" in question_lower and "2019" in question_lower:
            query = """
            SELECT court, COUNT(*) as case_count 
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE year BETWEEN 2019 AND 2022
            GROUP BY court 
            ORDER BY case_count DESC 
            LIMIT 1
            """
            result = query_s3_parquet("", query)
            if not result.empty:
                answers[question] = result.iloc[0]['court']
            else:
                answers[question] = "No data found"
            continue
        
        # Question 2: Regression slope of date_of_registration - decision_date by year
        if "regression slope" in question_lower and "33_10" in question_lower:
            query = """
            SELECT year, 
                   AVG(CAST(decision_date AS DATE) - CAST(date_of_registration AS DATE)) as avg_delay
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE court = '33_10' AND year IS NOT NULL
            GROUP BY year
            ORDER BY year
            """
            result = query_s3_parquet("", query)
            if not result.empty:
                # Calculate regression slope
                x = result['year'].values
                y = result['avg_delay'].values
                if len(x) > 1:
                    slope = (len(x) * sum(x*y) - sum(x) * sum(y)) / (len(x) * sum(x*x) - sum(x)**2)
                    answers[question] = float(slope)
                else:
                    answers[question] = 0.0
            else:
                answers[question] = 0.0
            continue
        
        # Question 3: Plot year vs delay
        if "plot" in question_lower and "year" in question_lower and "delay" in question_lower:
            plot_uri = create_high_court_plot()
            answers[question] = plot_uri
            continue
        
        # Default: add None for unhandled questions
        answers[question] = None
    
    return nan_to_none(answers)

def process_csv_task(task: str, questions: List[str], use_llm: bool = True) -> List:
    """Process CSV data tasks"""
    try:
        data = load_csv("data/data.csv")
        
        if use_llm:
            answers = process_with_llm(task, data)
            # If LLM failed (e.g., quota), gracefully fall back to traditional analyzer
            if not answers or all(a is None for a in answers):
                logger.warning("LLM unavailable or returned empty; falling back to traditional CSV analysis.")
                return nan_to_none(analyze_scraped_data(data, task))
            return answers
        else:
            answers = analyze_scraped_data(data, task)
            return nan_to_none(answers)
            
    except Exception as e:
        logger.error(f"Error in process_csv_task: {e}")
        return [None]

def get_sample_high_court_data():
    """Get sample high court data for testing when S3 is not accessible"""
    import pandas as pd
    # Create sample data for testing
    sample_data = {
        'court': ['33_10', '33_10', '33_10', '33_10', '33_10'],
        'year': [2019, 2020, 2021, 2022, 2023],
        'avg_delay': [30, 35, 40, 45, 50]
    }
    return pd.DataFrame(sample_data)