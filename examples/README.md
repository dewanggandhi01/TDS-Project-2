# LLM Integration Examples

This directory contains examples demonstrating the LLM integration capabilities of the Data Analyst Agent v3.0.

## Files

- `llm_examples.py` - Comprehensive examples of LLM-powered analysis

## Usage

### Prerequisites

1. **Start the server**:
   ```bash
   python run_server.py
   ```

2. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

3. **Run the examples**:
   ```bash
   python examples/llm_examples.py
   ```

## Example Queries

The examples demonstrate various types of natural language queries:

### Basic Analysis
- "How many movies are in the dataset?"
- "What is the average worldwide gross for all movies?"

### Statistical Analysis
- "Calculate the correlation between Rank and Peak columns"
- "Provide a comprehensive statistical summary"

### Data Filtering
- "How many movies grossed over $2 billion before 2020?"
- "Show movies released after 2010"

### Visualizations
- "Create a scatter plot of budget vs worldwide gross"
- "Show the trend of movie budgets over the years with a line chart"

### Complex Queries
- "Group movies by decade and calculate the average gross for each decade"
- "Create a histogram of worldwide gross values and overlay it with a normal distribution curve"

## Expected Output

The examples will show:
- ✅ Success/failure status for each query
- ⏱️ Response times for traditional vs LLM analysis
- 📊 Results including numbers, text, and base64-encoded plots
- 🔄 Comparison between traditional and LLM methods

## Troubleshooting

### Server Not Running
```
❌ Server is not running
   Start the server with: python run_server.py
```

### LLM Not Available
```
⚠️  LLM features are not available
   Set OPENAI_API_KEY environment variable to enable LLM features
```

### API Key Issues
```
❌ OpenAI API key test failed
   Check your API key at: https://platform.openai.com/api-keys
```

## Next Steps

After running the examples:
1. Try your own natural language queries
2. Explore the web interface at http://127.0.0.1:8000/
3. Check the API documentation at http://127.0.0.1:8000/docs
4. Read the main README.md for detailed usage instructions 