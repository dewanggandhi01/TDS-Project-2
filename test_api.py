import requests
import json
import time

def test_api():
    """Test the data analyst agent API"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/api/"
    
    # Test question file
    question_file = "data/question.txt"
    
    try:
        # Read the question file
        with open(question_file, 'r', encoding='utf-8') as f:
            question_content = f.read()
        
        print("Question content:")
        print(question_content)
        print("\n" + "="*50 + "\n")
        
        # Prepare the file for upload
        files = {
            'file': ('question.txt', question_content, 'text/plain')
        }
        
        print("Sending request to API...")
        start_time = time.time()
        
        # Make the request
        response = requests.post(url, files=files, timeout=180)  # 3 minutes timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response format
            if isinstance(result, list) and len(result) == 4:
                print("\n✅ Response format is correct (4-element array)")
                
                # Check each answer
                for i, answer in enumerate(result):
                    if answer is not None:
                        print(f"✅ Answer {i+1}: {type(answer).__name__} = {answer}")
                    else:
                        print(f"⚠️  Answer {i+1}: None")
            else:
                print("\n❌ Response format is incorrect")
                
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (took longer than 3 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the server is running on http://127.0.0.1:8000")
    except FileNotFoundError:
        print(f"❌ Question file not found: {question_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(response.json())
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("Testing Data Analyst Agent API")
    print("="*50)
    
    # Test health endpoint first
    test_health_endpoint()
    print()
    
    # Test main API
    test_api() 