#!/usr/bin/env python3
"""
LLM Integration Examples for Data Analyst Agent v3.0
Demonstrates various natural language analysis capabilities
"""

import requests
import json
import time
import os

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_llm_analysis():
    """Test LLM-powered analysis with various queries"""
    
    # Test data
    test_cases = [
        {
            "name": "Basic Counting",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "How many movies are in the dataset?"
        },
        {
            "name": "Statistical Analysis",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "What is the average worldwide gross for all movies?"
        },
        {
            "name": "Correlation Analysis",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "Calculate the correlation between Rank and Peak columns"
        },
        {
            "name": "Data Filtering",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "How many movies grossed over $2 billion before 2020?"
        },
        {
            "name": "Time Series Analysis",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "Show the trend of movie budgets over the years with a line chart"
        },
        {
            "name": "Advanced Aggregation",
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "task": "Group movies by decade and calculate the average gross for each decade"
        }
    ]
    
    print("🤖 LLM Analysis Examples")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Query: {test_case['task']}")
        
        try:
            # Make LLM request
            response = requests.post(
                f"{BASE_URL}/api/llm/analyze",
                json={
                    "url": test_case["url"],
                    "task": test_case["task"]
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("   ✅ Success")
                    print(f"   Method: {result.get('method')}")
                    
                    # Display result
                    if isinstance(result.get('result'), list):
                        for j, answer in enumerate(result['result']):
                            if answer is not None:
                                print(f"   Answer {j+1}: {answer}")
                    else:
                        print(f"   Result: {result.get('result')}")
                else:
                    print("   ❌ Failed")
                    print(f"   Error: {result.get('error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Small delay between requests
        time.sleep(1)

def test_traditional_vs_llm():
    """Compare traditional vs LLM analysis"""
    
    print("\n🔄 Traditional vs LLM Comparison")
    print("=" * 50)
    
    test_query = {
        "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
        "task": "How many movies grossed over $2 billion before 2020?"
    }
    
    # Test traditional analysis
    print("\n1. Traditional Analysis:")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/traditional/analyze",
            json=test_query,
            timeout=180
        )
        traditional_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success (Time: {traditional_time:.2f}s)")
            print(f"   Method: {result.get('method')}")
            print(f"   Result: {result.get('result')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test LLM analysis
    print("\n2. LLM Analysis:")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/llm/analyze",
            json=test_query,
            timeout=180
        )
        llm_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success (Time: {llm_time:.2f}s)")
            print(f"   Method: {result.get('method')}")
            print(f"   Result: {result.get('result')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def test_complex_queries():
    """Test complex natural language queries"""
    
    print("\n🧠 Complex Query Examples")
    print("=" * 50)
    
    complex_queries = [
        {
            "name": "Multi-step Analysis",
            "task": "First, filter movies released after 2010, then calculate the correlation between budget and worldwide gross, and finally create a scatter plot"
        },
        {
            "name": "Conditional Analysis",
            "task": "If there are more than 50 movies in the dataset, show the top 10 by worldwide gross, otherwise show all movies"
        },
        {
            "name": "Statistical Summary",
            "task": "Provide a comprehensive statistical summary including mean, median, standard deviation, and quartiles for all numeric columns"
        },
        {
            "name": "Data Visualization",
            "task": "Create a histogram of worldwide gross values and overlay it with a normal distribution curve"
        }
    ]
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n{i}. {query['name']}")
        print(f"   Query: {query['task']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/llm/analyze",
                json={
                    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
                    "task": query["task"]
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("   ✅ Success")
                    print(f"   Method: {result.get('method')}")
                    
                    # Check if result contains a plot
                    if result.get('result') and isinstance(result['result'], str) and result['result'].startswith('data:image'):
                        print("   📊 Plot generated")
                    else:
                        print(f"   Result: {result.get('result')}")
                else:
                    print("   ❌ Failed")
                    print(f"   Error: {result.get('error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        time.sleep(1)

def check_server_status():
    """Check if the server is running and LLM is available"""
    
    print("🔍 Checking Server Status")
    print("=" * 50)
    
    try:
        # Check health endpoint
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is running")
            print(f"   Service: {data.get('service')}")
            print(f"   LLM Available: {data.get('llm_available')}")
            
            if data.get('llm_available'):
                print("   🤖 LLM features are enabled")
            else:
                print("   ⚠️  LLM features are disabled (no API key)")
            
            return data.get('llm_available', False)
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running")
        print("   Start the server with: python run_server.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

def main():
    """Main function to run all examples"""
    
    print("🚀 Data Analyst Agent v3.0 - LLM Examples")
    print("=" * 60)
    
    # Check server status
    llm_available = check_server_status()
    
    if not llm_available:
        print("\n⚠️  LLM features are not available")
        print("   Set OPENAI_API_KEY environment variable to enable LLM features")
        print("   Traditional analysis examples will still work")
    
    # Run examples
    test_llm_analysis()
    test_traditional_vs_llm()
    test_complex_queries()
    
    print("\n🎉 Examples completed!")
    print("\n📚 For more information:")
    print("   - API Documentation: http://127.0.0.1:8000/docs")
    print("   - Web Interface: http://127.0.0.1:8000/")
    print("   - README.md for detailed usage instructions")

if __name__ == "__main__":
    main() 