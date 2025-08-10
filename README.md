# 🧠🌐 **[Live Demo](https://tds-project-2-git-main-dewanggandhi01s-projects.vercel.app)** | 📖 **[API Docs](https://tds-project-2-git-main-dewanggandhi01s-projects.vercel.app/docs)**LLM Data Analyst Agent

A powerful web-based data analysis platform that uses **Large Language Models** and ReAct (Reasoning and Acting) agents to source, prepare, analyze, and visualize any data. This application provides both a modern web interface and a robust API for complex data analysis tasks including web scraping, statistical analysis, and data visualization.

� **[Live Demo](https://your-vercel-deployment.vercel.app)** | 📖 **[API Docs](https://your-vercel-deployment.vercel.app/docs)**

## ✨ Features

### 🤖 LLM-Powered Analysis
- **Natural Language Understanding**: Ask questions in plain English
- **Dynamic Code Generation**: LLM writes Python code for your specific query
- **Safe Execution**: Code runs in restricted environment with only safe modules
- **Intelligent Fallback**: Automatically falls back to traditional methods if LLM fails

### 🌐 Web Interface & API
- **Modern Web Interface**: Beautiful, responsive UI built with Bootstrap and JavaScript
- **Real-time Results**: See analysis results with visualizations in real-time
- **Multiple Data Sources**: Support for Wikipedia, S3 parquet files, and CSV data
- **FastAPI**: Modern, fast web framework with automatic API documentation

### 📊 Data Analysis Capabilities
- **Web Scraping**: Automatic data extraction from Wikipedia and other sources
- **Statistical Analysis**: Correlation analysis, regression, and more
- **Data Visualization**: Automatic plot generation with matplotlib
- **Large Dataset Support**: Efficient processing with DuckDB

## 🚀 Quick Start

### 🌐 Online Demo
Visit our [live demo](https://your-vercel-deployment.vercel.app) to try the application immediately!

### 🏠 Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/dewanggandhi01/TDS-Project-2.git
   cd TDS-Project-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key
   ```

4. **Start the server**
   ```bash
   python run_server.py
   ```

5. **Open your browser**
   - Web Interface: `http://127.0.0.1:8000/`
   - API Documentation: `http://127.0.0.1:8000/docs`

## 💡 Usage Examples

### Web Interface
1. Enter a data source URL (e.g., Wikipedia page)
2. Add analysis tasks using natural language
3. Choose LLM-powered or Traditional analysis
4. View results with interactive visualizations

### API Usage
```bash
# LLM-Powered Analysis
curl -X POST "https://your-app.vercel.app/api/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
    "task": "How many movies grossed over $2 billion before 2020?"
  }'

# Traditional Analysis
curl -X POST "https://your-app.vercel.app/api/traditional/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
    "task": "Show correlation between rank and worldwide gross"
  }'
```

## 🚀 Deployment

### Vercel (Recommended)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/dewanggandhi01/TDS-Project-2)

1. **One-click deploy**: Click the button above
2. **Set environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: Set to `production`
3. **Deploy**: Vercel will automatically build and deploy your app

### Manual Vercel Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Set environment variables
vercel env add OPENAI_API_KEY
vercel env add ENVIRONMENT production
```

### Other Platforms

#### Railway
```bash
railway login
railway init
railway up
# Set environment variables in Railway dashboard
```

#### Heroku
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
heroku config:set ENVIRONMENT=production
git push heroku main
```

## 🛠️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM features)
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `ENVIRONMENT`: Set to `production` for deployment
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## 🏗️ Architecture

```
TDS-Project-2/
├── app/                        # Core application logic
│   ├── main.py                # FastAPI application with LLM endpoints
│   ├── agent.py               # Main agent orchestrator
│   ├── llm_agent.py           # LLM-based task execution
│   ├── data_loader.py         # Data loading utilities
│   ├── analyzer.py            # Data analysis functions
│   └── visualizer.py          # Plot generation
├── templates/                  # HTML templates
├── static/                     # Static files (CSS, JS, images)
├── requirements.txt            # Python dependencies
├── vercel.json                # Vercel deployment config
├── run_server.py              # Local development server
└── README.md                  # This file
```

## 🔒 Security

### LLM Code Execution Safety
- **Restricted Modules**: Only pandas, numpy, matplotlib allowed
- **No File Access**: Cannot read/write files
- **No Network Access**: Cannot make external requests
- **Memory Limits**: Prevents memory exhaustion
- **Timeout Protection**: Prevents infinite loops

## 🧪 API Reference

### Health Check
```bash
GET /health
```

### LLM Analysis
```bash
POST /api/llm/analyze
{
  "url": "https://example.com/data",
  "task": "Analyze this data and find trends"
}
```

### Traditional Analysis
```bash
POST /api/traditional/analyze
{
  "url": "https://example.com/data", 
  "task": "Generate basic statistics"
}
```

For complete API documentation, visit `/docs` on your deployed application.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Acknowledgments

- [OpenAI](https://openai.com) for providing GPT-4 API
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Vercel](https://vercel.com) for seamless deployment platform

---

**⭐ Star this repository if you find it helpful!**
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

4. **Set up OpenAI API Key**
   ```bash
   # Option 1: Environment variable
   export OPENAI_API_KEY="your_openai_api_key_here"
   
   # Option 2: Create .env file (copy from env_example.txt)
   cp env_example.txt .env
   # Edit .env and add your API key
   ```

## 🚀 Quick Start

### Local Development

1. **Start the server**
   ```bash
   python run_server.py
   ```

2. **Access the web interface**
   - Web Interface: `http://127.0.0.1:8000/`
   - API Documentation: `http://127.0.0.1:8000/docs`
   - Health check: `http://127.0.0.1:8000/health`
   - Configuration: `http://127.0.0.1:8000/api/config`

3. **Test the API**
   ```bash
   python test_api.py
   ```

### Using the Web Interface

1. **Open the web interface** at `http://127.0.0.1:8000/`
2. **Enter a data source URL** (optional) - e.g., Wikipedia page URL
3. **Add analysis tasks** using natural language
4. **Choose analysis mode**: LLM-powered or Traditional
5. **Click "Analyze Data"** to process your tasks
6. **View results** with visualizations and statistics

### Using the API

#### LLM-Powered Analysis (Recommended)
```bash
curl -X POST "http://127.0.0.1:8000/api/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
    "task": "How many movies grossed over $2 billion before 2020?"
  }'
```

#### Traditional Analysis
```bash
curl -X POST "http://127.0.0.1:8000/api/traditional/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
    "task": "How many movies grossed over $2 billion before 2020?"
  }'
```

#### File Upload with LLM
```bash
curl "http://127.0.0.1:8000/api/" \
  -F "file=@data/question.txt" \
  -F "use_llm=true"
```

## 🏗️ Architecture

### LLM Integration Flow

1. **User Input**: Natural language query + dataset URL
2. **Data Fetching**: Scrape data from URL using BeautifulSoup
3. **Data Analysis**: LLM generates Python code for the query
4. **Safe Execution**: Code runs in restricted environment
5. **Result Processing**: Extract results and visualizations
6. **Response**: Return structured results to user

### LLM Agent Implementation

The system uses OpenAI GPT-4 for intelligent task execution:

```python
# Example LLM prompt
system_prompt = """
You are a data analysis expert. Given a dataset and a natural language query, 
generate executable Python code to solve the query.

IMPORTANT RULES:
1. Use ONLY these modules: pandas (as pd), numpy (as np), matplotlib.pyplot (as plt)
2. The dataset is available as a pandas DataFrame called 'df'
3. For visualizations, save plots to a BytesIO buffer and return as base64 string
4. Return ONLY the Python code, no explanations
5. Handle errors gracefully
"""
```

### Safe Code Execution

Generated code runs in a restricted environment:
- Only safe modules allowed (pandas, numpy, matplotlib)
- No file system access
- No network access
- Memory limits enforced
- Timeout protection

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

### LLM-Powered Analysis Examples

```python
# Complex statistical analysis
task = "Calculate the correlation between movie budget and worldwide gross, and create a scatter plot"

# Time series analysis
task = "Show the trend of movie budgets over the last 20 years with a line chart"

# Advanced filtering
task = "Find all movies that grossed over $1 billion and were released in summer months"

# Custom aggregations
task = "Group movies by decade and calculate average gross for each decade"
```

### Traditional Analysis (Still Supported)
```python
# Pre-defined questions for Wikipedia film data
questions = [
    "How many $2 bn movies were released before 2020?",
    "Which is the earliest film that grossed over $1.5 bn?",
    "What's the correlation between the Rank and Peak?",
    "Draw a scatterplot of Rank and Peak with regression line."
]
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM features)
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `OPENAI_MAX_TOKENS`: Maximum tokens for code generation (default: 2000)
- `OPENAI_TEMPERATURE`: Creativity level (default: 0.1)
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 127.0.0.1)
- `LOG_LEVEL`: Logging level (default: info)

### API Configuration
- Timeout: 3 minutes for analysis tasks
- File size limit: 10MB for uploaded files
- Supported formats: .txt files
- LLM fallback: Automatic fallback to traditional methods

## 📈 Performance

- **LLM Response Time**: 10-30 seconds for complex queries
- **Traditional Response Time**: 2-5 seconds for predefined tasks
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Supports concurrent requests
- **Error Handling**: Comprehensive error handling and logging

## 🧪 Testing

### Run All Tests
```bash
python test_api.py
```

### Test Specific Endpoints
```bash
# Test health
curl http://127.0.0.1:8000/health

# Test LLM endpoint
curl -X POST "http://127.0.0.1:8000/api/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{"task": "Count rows in dataset"}'

# Test traditional endpoint
curl -X POST "http://127.0.0.1:8000/api/traditional/analyze" \
  -H "Content-Type: application/json" \
  -d '{"task": "Count rows in dataset"}'
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
4. Set environment variables: `railway variables set OPENAI_API_KEY=your_key`
5. Deploy: `railway up`

#### Heroku
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Set environment variables: `heroku config:set OPENAI_API_KEY=your_key`
5. Deploy: `git push heroku main`

#### Render
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python run_server.py`
6. Add environment variable: `OPENAI_API_KEY`

## 📁 Project Structure

```
TDS-Project-2/
├── app/                        # Core application logic
│   ├── __init__.py
│   ├── main.py                # FastAPI application with LLM endpoints
│   ├── agent.py               # Main agent orchestrator with LLM integration
│   ├── llm_agent.py           # LLM-based task execution
│   ├── react_agent.py         # ReAct agent implementation
│   ├── data_loader.py         # Data loading utilities
│   ├── analyzer.py            # Data analysis functions
│   ├── visualizer.py          # Plot generation
│   └── utils.py               # Utility functions
├── templates/                  # HTML templates
├── static/                     # Static files (CSS, JS, images)
├── data/                       # Data files
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── run_server.py              # Server startup script
├── test_api.py                # API testing script
├── env_example.txt            # Environment variables example
├── Procfile                   # Heroku deployment
├── runtime.txt                # Python version specification
└── README.md                  # This file
```

## 🔒 Security

### LLM Code Execution Safety
- **Restricted Modules**: Only pandas, numpy, matplotlib allowed
- **No File Access**: Cannot read/write files
- **No Network Access**: Cannot make external requests
- **Memory Limits**: Prevents memory exhaustion
- **Timeout Protection**: Prevents infinite loops
- **Error Handling**: Graceful failure handling

### API Security
- **Input Validation**: All inputs validated and sanitized
- **Rate Limiting**: Built-in rate limiting for API calls
- **Error Sanitization**: No sensitive information in error messages
- **CORS Configuration**: Proper CORS setup for web interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing GPT-4 API
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
- Check the health endpoint for system status

---

**Note**: This API now supports both LLM-powered and traditional analysis. The LLM features require an OpenAI API key and may incur costs based on usage. Traditional analysis is always available as a fallback option.

