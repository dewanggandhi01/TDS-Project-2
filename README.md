# Data Analyst Agent API

A powerful data analysis API that uses ReAct (Reasoning and Acting) agents to source, prepare, analyze, and visualize any data. This API can handle complex data analysis tasks including web scraping, statistical analysis, and data visualization.

## 🚀 Features

- **ReAct Agent**: Implements Reasoning and Acting pattern for intelligent data analysis
- **Web Scraping**: Automatically scrape data from Wikipedia and other websites
- **Data Analysis**: Perform statistical analysis, correlations, and aggregations
- **Data Visualization**: Generate scatterplots, regression lines, and other visualizations
- **Multiple Data Sources**: Support for Wikipedia, S3 parquet files, and CSV data
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Base64 Image Encoding**: Return plots as base64 data URIs for easy integration

## 📋 Requirements

- Python 3.8+
- FastAPI
- Pandas
- NumPy
- Matplotlib
- BeautifulSoup4
- DuckDB
- Requests

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dewanggandhi01/TDS-Project-2.git
   cd TDS-Project-2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Local Development

1. **Start the server**
   ```bash
   python run_server.py
   ```

2. **Test the API**
   ```bash
   python test_api.py
   ```

3. **Access the API**
   - API endpoint: `http://127.0.0.1:8000/api/`
   - Health check: `http://127.0.0.1:8000/health`
   - API docs: `http://127.0.0.1:8000/docs`

### Using the API

Send a POST request to `/api/` with a text file containing your analysis question:

```bash
curl "http://127.0.0.1:8000/api/" -F "@data/question.txt"
```

Example question file (`data/question.txt`):
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, "data:image/png;base64,iVBORw0KG..." under 100,000 bytes.
```

## 🏗️ Architecture

### ReAct Agent Implementation

The system uses a ReAct (Reasoning and Acting) agent pattern:

1. **Think**: Analyze the task and plan actions
2. **Act**: Execute planned actions using available tools
3. **Observe**: Process results and extract answers

### Core Components

- **`app/agent.py`**: Main agent orchestrator
- **`app/react_agent.py`**: ReAct agent implementation
- **`app/data_loader.py`**: Data loading and scraping utilities
- **`app/analyzer.py`**: Data analysis functions
- **`app/visualizer.py`**: Plot generation and visualization
- **`app/main.py`**: FastAPI application

### Available Tools

- `scrape_wikipedia`: Scrape data from Wikipedia tables
- `analyze_data`: Perform statistical analysis
- `create_plot`: Generate visualizations
- `query_s3`: Query S3 parquet files
- `calculate_statistics`: Basic statistical operations

## 📊 Supported Data Sources

### 1. Wikipedia Data
- Automatically scrape tables from Wikipedia pages
- Clean and process scraped data
- Handle various table formats

### 2. S3 Parquet Files
- Query large datasets using DuckDB
- Support for partitioned data
- Efficient data processing

### 3. CSV Files
- Local CSV file processing
- Basic data analysis operations

## 🎯 Example Use Cases

### Film Analysis
```python
# Analyze highest-grossing films
question = """
Scrape the list of highest grossing films from Wikipedia.
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak with regression line.
"""
```

### High Court Data Analysis
```python
# Analyze Indian High Court judgments
question = """
The Indian high court judgement dataset contains judgments from the Indian High Courts.
1. Which high court disposed the most cases from 2019-2022?
2. What's the regression slope of the date_of_registration - decision_date by year in court=33_10?
3. Plot the year and # of days of delay as a scatterplot with regression line.
"""
```

## 🚀 Deployment

### Local Development
```bash
python run_server.py
```

### Production Deployment

#### Railway (Recommended)
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

#### Heroku
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

#### Render
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python run_server.py`

#### Using ngrok (for testing)
```bash
ngrok http 8000
```

## 🧪 Testing

### Run Tests
```bash
python test_api.py
```

### Manual Testing
```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test main API
curl "http://127.0.0.1:8000/api/" -F "@data/question.txt"
```

## 📁 Project Structure

```
TDS-Project-2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── agent.py             # Main agent orchestrator
│   ├── react_agent.py       # ReAct agent implementation
│   ├── data_loader.py       # Data loading utilities
│   ├── analyzer.py          # Data analysis functions
│   ├── visualizer.py        # Plot generation
│   └── utils.py             # Utility functions
├── data/
│   ├── question.txt         # Sample question file
│   └── data.csv            # Sample CSV data
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
├── run_server.py           # Server startup script
├── test_api.py             # API testing script
├── deploy.py               # Deployment script
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 127.0.0.1)
- `LOG_LEVEL`: Logging level (default: info)

### API Configuration
- Timeout: 3 minutes for analysis tasks
- File size limit: 10MB for uploaded files
- Supported formats: .txt files

## 📈 Performance

- **Response Time**: Typically under 3 minutes for complex analyses
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Supports concurrent requests
- **Error Handling**: Comprehensive error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Pandas for data manipulation
- Matplotlib for visualization
- BeautifulSoup for web scraping
- DuckDB for efficient data querying

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the test files for usage examples

---

**Note**: This API is designed to handle complex data analysis tasks and may take up to 3 minutes to process requests. The system automatically handles data scraping, cleaning, analysis, and visualization based on the provided questions.

