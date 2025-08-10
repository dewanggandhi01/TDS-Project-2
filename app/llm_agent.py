import os
import json
import re
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
import logging
import time

logger = logging.getLogger(__name__)

# Simple in-process quota cooldown to avoid repeated 429 spam
_QUOTA_BLOCK_UNTIL_TS: float = 0.0

def _quota_block_active() -> bool:
    return time.time() < _QUOTA_BLOCK_UNTIL_TS

def _set_quota_block(seconds: int = 600) -> None:
    global _QUOTA_BLOCK_UNTIL_TS
    _QUOTA_BLOCK_UNTIL_TS = time.time() + max(1, int(seconds))

class LLMAgent:
    """LLM-based agent for natural language task execution"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM agent"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Optional overrides for proxies or gateways
        base_url = os.getenv('OPENAI_BASE_URL')  # e.g., https://api.openai.com/v1 or a proxy URL
        organization = os.getenv('OPENAI_ORG')

        # Azure OpenAI support
        use_azure = os.getenv('OPENAI_API_TYPE', '').lower() in { 'azure', 'azure_openai' } or bool(os.getenv('AZURE_OPENAI_ENDPOINT'))
        if use_azure and AzureOpenAI is not None:
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') or base_url
            api_version = os.getenv('OPENAI_API_VERSION', '2024-06-01')
            if not azure_endpoint:
                raise ValueError("For Azure OpenAI, set AZURE_OPENAI_ENDPOINT or OPENAI_BASE_URL to your resource endpoint.")
            self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=azure_endpoint, api_version=api_version)
        else:
            # Basic sanity check for OpenAI keys when using default base URL
            if not base_url and not self.api_key.startswith("sk-"):
                # Likely an Azure or non-OpenAI key being used against api.openai.com
                raise ValueError(
                    "OPENAI_API_KEY doesn't look like an OpenAI key. Set OPENAI_BASE_URL for a gateway or "
                    "use Azure settings (OPENAI_API_TYPE=azure and AZURE_OPENAI_ENDPOINT)."
                )
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, organization=organization)
        self.safe_modules = {
            'pandas': pd, 'pd': pd,
            'numpy': np, 'np': np,
            'matplotlib.pyplot': plt, 'plt': plt,
            'base64': base64,
            'io': io
        }
    
    def generate_code(self, task: str, data_info: Dict[str, Any]) -> str:
        """Generate Python code for the given task using LLM"""
        
        # Short-circuit if we've recently hit quota issues
        if _quota_block_active():
            raise RuntimeError("LLM_QUOTA_EXCEEDED: Temporary cooldown active due to prior 429/insufficient_quota.")

        system_prompt = """You are a data analysis expert. Given a dataset and a natural language query, generate executable Python code to solve the query.

IMPORTANT RULES:
1. Use ONLY these modules: pandas (as pd), numpy (as np), matplotlib.pyplot (as plt), base64, io
2. The dataset is available as a pandas DataFrame called 'df'
3. For visualizations, save plots to a BytesIO buffer and return as base64 string
4. Return ONLY the Python code, no explanations
5. Handle errors gracefully
6. For plots, use plt.figure() and plt.close() to manage memory
7. Return results in a format that can be easily converted to JSON

DATASET INFO:
{data_info}

USER QUERY: {task}

Generate Python code that:
1. Analyzes the data according to the query
2. Returns results as a dictionary with keys: 'result', 'type', 'plot' (if applicable)
3. For plots, include 'plot' key with base64 encoded image
4. For numerical/text results, include 'result' key with the answer
5. Include 'type' key indicating result type: 'text', 'number', 'list', 'plot'"""

        try:
            # Support both OPENAI_MODEL and MODEL env vars; prefer a lighter default
            model = os.getenv('OPENAI_MODEL') or os.getenv('MODEL', 'gpt-4o-mini')
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt.format(
                        data_info=json.dumps(data_info, indent=2),
                        task=task
                    )},
                    {"role": "user", "content": f"Generate code for: {task}"}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up the code (remove markdown if present)
            if code.startswith('```python'):
                code = code[9:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            msg = str(e)
            # Detect 429 / insufficient quota and engage cooldown to reduce noise
            if 'insufficient_quota' in msg or 'Error code: 429' in msg or 'status_code=429' in msg:
                logger.error("OpenAI quota exceeded (429). Enabling temporary cooldown and signaling fallback.")
                _set_quota_block(600)
                raise RuntimeError("LLM_QUOTA_EXCEEDED: Insufficient quota or rate limit.")
            logger.error(f"Error generating code: {e}")
            raise
    
    def execute_code_safely(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the generated code in a safe environment"""
        
        # Create a restricted namespace
        namespace = {
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'plt': plt,
            'base64': base64,
            'io': io,
            'BytesIO': io.BytesIO,
            'json': json
        }
        
        try:
            # Execute the code
            exec(code, namespace)
            
            # Look for result in namespace
            result = None
            result_type = 'text'
            plot_data = None
            
            # Check if result was returned
            if 'result' in namespace:
                result = namespace['result']
            elif 'output' in namespace:
                result = namespace['output']
            else:
                # Look for common variable names
                for var in ['answer', 'count', 'correlation', 'mean', 'median', 'sum']:
                    if var in namespace:
                        result = namespace[var]
                        break
            
            # Check for plot
            if 'plot' in namespace:
                plot_data = namespace['plot']
                result_type = 'plot'
            elif 'plot_base64' in namespace:
                plot_data = namespace['plot_base64']
                result_type = 'plot'
            
            # Determine result type
            if result is not None:
                if isinstance(result, (int, float)):
                    result_type = 'number'
                elif isinstance(result, list):
                    result_type = 'list'
                elif isinstance(result, str):
                    result_type = 'text'
            
            return {
                'success': True,
                'result': result,
                'type': result_type,
                'plot': plot_data,
                'code': code
            }
            
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': code
            }
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract information about the dataset for the LLM"""
        info = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(3).to_dict('records'),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Add some statistics for numeric columns
        if len(info['numeric_columns']) > 0:
            info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
        
        return info
    
    def process_task(self, task: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Main method to process a task using LLM"""
        try:
            # Get dataset information
            data_info = self.get_data_info(df)
            
            # Generate code
            code = self.generate_code(task, data_info)
            
            # Execute code safely
            result = self.execute_code_safely(code, df)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_task: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Global LLM agent instance
llm_agent = None

def get_llm_agent() -> LLMAgent:
    """Get or create LLM agent instance"""
    global llm_agent
    if llm_agent is None:
        llm_agent = LLMAgent()
    return llm_agent 