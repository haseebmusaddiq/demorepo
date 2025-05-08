from typing import List, Dict, Any, Optional, Union
import yaml
import traceback
import json
import os
import time
import requests
import gc
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

class LLMManager:
    def __init__(self, config_path_or_dict):
        print("[DEBUG] Initializing LLMManager")
        try:
            # Handle either a config dict or a path to a config file
            if isinstance(config_path_or_dict, dict):
                self.config = config_path_or_dict
                print("[DEBUG] Using provided config dictionary")
            else:
                # Assume it's a path to a config file
                with open(config_path_or_dict, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"[DEBUG] Config loaded from file: {config_path_or_dict}")
            
            # Set default provider to 'tinyllama' if not specified
            if 'llm' not in self.config:
                self.config['llm'] = {}
            if 'provider' not in self.config['llm']:
                self.config['llm']['provider'] = 'tinyllama'
                print("[DEBUG] No provider specified, defaulting to 'tinyllama'")
            
            # Load TinyLlama model from Hugging Face
            print("[INFO] Loading TinyLlama model from Hugging Face")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # Use TinyLlama model suitable for 8GB RAM
                small_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"[DEBUG] Loading model: {small_model_name}")
                
                self.local_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    small_model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                print(f"[DEBUG] Successfully loaded model: {small_model_name}")
            except Exception as model_error:
                print(f"[ERROR] Failed to load model: {model_error}")
                traceback.print_exc()
                raise
            
            # Initialize other providers if configured
            if 'anthropic' in self.config:
                print("[DEBUG] Anthropic configuration found")
            
            if 'local' in self.config:
                print("[DEBUG] Local LLM configuration found")
                
            print("[DEBUG] LLMManager initialized successfully")
        except Exception as e:
            print(f"[ERROR] Error initializing LLMManager: {e}")
            traceback.print_exc()
            raise

    def build_prompt(self, query: str, contexts: List[str], max_context_length: int = 3000) -> str:
        """
        Build a prompt for the LLM with the query and context.
        
        Args:
            query: The user's query
            contexts: List of relevant context texts
            max_context_length: Maximum length for the context
            
        Returns:
            Formatted prompt string
        """
        print(f"[DEBUG] Building prompt for query: {query[:50]}...")
        print(f"[DEBUG] Number of context chunks: {len(contexts)}")
        
        try:
            # Join contexts and truncate if too long
            context = "\n\n".join(contexts)
            original_context_length = len(context)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
                print(f"[DEBUG] Context truncated from {original_context_length} to {max_context_length} characters")
            else:
                print(f"[DEBUG] Context length: {len(context)} characters (under limit)")
            
            # Check if this is a code generation request
            is_code_request = any(keyword in query.lower() for keyword in 
                                ['code', 'function', 'script', 'program', 'implement', 'write'])
            
            print(f"[DEBUG] Is code request: {is_code_request}")
            
            if is_code_request:
                prompt = f"""Based on the following context, please generate code that addresses the request.

Context:
{context}

Request: {query}

Please provide your answer as code with proper formatting using markdown code blocks with the appropriate language.
"""
            else:
                prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}
Answer:"""
                    
            # Print the full prompt for debugging
            print("\n=== FINAL PROMPT (FULL) ===")
            print(prompt)
            print(f"=== END OF PROMPT (Total length: {len(prompt)} characters) ===\n")
            
            # Also print a truncated version for the logs
            print("\n=== Generated Prompt (Truncated) ===")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print(f"=== (Total prompt length: {len(prompt)} characters) ===\n")
            
            return prompt
        except Exception as e:
            print(f"[ERROR] Error in build_prompt: {e}")
            traceback.print_exc()
            # Return a simplified prompt as fallback
            return f"Please answer this question or generate code for this request: {query}"
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM with retry logic for transient errors"""
        print(f"[DEBUG] Generating response for prompt of length {len(prompt)}")
        
        # Try different providers in sequence
        providers_to_try = []
        
        # Get the configured provider
        provider = self.config.get('llm', {}).get('provider', 'tinyllama').lower()
        
        # Add the configured provider first
        if provider == 'anthropic':
            providers_to_try.append(('anthropic', self._generate_anthropic_response))
        elif provider == 'local_api':
            providers_to_try.append(('local_api', self._generate_local_api_response))
        else:  # Default to TinyLlama
            providers_to_try.append(('tinyllama', self._generate_tinyllama_response))
        
        # Add fallback providers
        if provider != 'local_api':
            providers_to_try.append(('local_api', self._generate_local_api_response))
        if provider != 'anthropic':
            providers_to_try.append(('anthropic', self._generate_anthropic_response))
        if provider != 'tinyllama':
            providers_to_try.append(('tinyllama', self._generate_tinyllama_response))
        
        # Try each provider in sequence
        last_error = None
        for provider_name, provider_func in providers_to_try:
            try:
                print(f"[DEBUG] Trying provider: {provider_name}")
                response = provider_func(prompt)
                if response and len(response) > 20:  # Ensure we got a meaningful response
                    return response
                else:
                    print(f"[WARNING] Provider {provider_name} returned empty or too short response, trying next")
            except Exception as e:
                print(f"[WARNING] Provider {provider_name} failed: {e}")
                last_error = e
                continue
        
        # If all providers failed, return a simple error message
        error_msg = str(last_error) if last_error else "Unknown error"
        print(f"[ERROR] All providers failed. Last error: {error_msg}")
        
        return "I encountered an error while generating a response. Please try a different question or try again later."

    def _generate_anthropic_response(self, prompt: str) -> str:
        """Generate a response using Anthropic API"""
        print(f"[DEBUG] Generating Anthropic response for prompt of length {len(prompt)}")
        try:
            if 'anthropic' not in self.config:
                raise ValueError("Anthropic configuration not found in config file")
            
            api_key = self.config['anthropic'].get('api_key')
            if not api_key:
                raise ValueError("Anthropic API key not found in config")
            
            model = self.config['anthropic'].get('model', 'claude-2')
            max_tokens = self.config['anthropic'].get('max_tokens', 1000)
            temperature = self.config['anthropic'].get('temperature', 0.7)
            
            print(f"[DEBUG] Using Anthropic model: {model}, max_tokens: {max_tokens}, temperature: {temperature}")
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": model,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                raise ValueError(f"Anthropic API returned status code {response.status_code}: {response.text}")
            
            result = response.json().get("completion", "")
            print(f"[DEBUG] Anthropic response received, length: {len(result)}")
            return result
        except Exception as e:
            print(f"[ERROR] Error in _generate_anthropic_response: {e}")
            traceback.print_exc()
            raise

    def _generate_local_api_response(self, prompt: str) -> str:
        """Generate a response using a local API endpoint"""
        print(f"[DEBUG] Generating local API response for prompt of length {len(prompt)}")
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
            reraise=True
        )
        def _call_api_with_retry():
            endpoint = self.config.get('local', {}).get('endpoint', 'http://localhost:8000/v1/completions')
            model = self.config.get('local', {}).get('model', 'local-model')
            max_tokens = self.config.get('local', {}).get('max_tokens', 1000)
            temperature = self.config.get('local', {}).get('temperature', 0.7)
            
            print(f"[DEBUG] Using local API at endpoint: {endpoint}, max_tokens: {max_tokens}, temperature: {temperature}")
            
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                raise ValueError(f"Local API returned status code {response.status_code}: {response.text}")
            
            result = response.json().get("choices", [{}])[0].get("text", "")
            print(f"[DEBUG] Local API response received, length: {len(result)}")
            return result
        
        try:
            return _call_api_with_retry()
        except RetryError as e:
            print(f"[ERROR] Max retries exceeded for local API: {e}")
            raise ValueError(f"Failed to connect to local API after multiple attempts: {str(e.__cause__)}")
        except Exception as e:
            print(f"[ERROR] Error in _generate_local_api_response: {e}")
            traceback.print_exc()
            raise

    def _generate_tinyllama_response(self, prompt: str) -> str:
        """Generate a response using the local TinyLlama model"""
        print(f"[DEBUG] Generating TinyLlama response for prompt of length {len(prompt)}")
        
        # Import required libraries at the module level
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            # Check if the model is loaded
            if not hasattr(self, 'local_model') or self.local_model is None or not hasattr(self, 'local_tokenizer') or self.local_tokenizer is None:
                print("[WARNING] TinyLlama model or tokenizer not properly initialized, attempting to reload")
                
                # Load the model and tokenizer
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"[DEBUG] Loading model: {model_name}")
                
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                print(f"[DEBUG] Successfully reloaded model: {model_name}")
            
            # Tokenize the prompt
            inputs = self.local_tokenizer(prompt, return_tensors="pt")
            
            # Generate with parameters from config
            max_new_tokens = self.config.get('llm', {}).get('max_new_tokens', 200)
            temperature = self.config.get('llm', {}).get('temperature', 0.7)
            
            print(f"[DEBUG] Generating with max_new_tokens={max_new_tokens}, temperature={temperature}")
            
            with torch.no_grad():
                output = self.local_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.92,
                    repetition_penalty=1.2
                )
            
            # Decode the generated text
            response = self.local_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the generated part, not the prompt
            prompt_text = self.local_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            response_only = response[len(prompt_text):].strip()
            
            print(f"[DEBUG] TinyLlama response generated, length: {len(response_only)}")
            return response_only
            
        except Exception as e:
            print(f"[ERROR] Error in _generate_tinyllama_response: {e}")
            traceback.print_exc()
            
            # Try to free up memory
            if hasattr(self, 'local_model'):
                del self.local_model
            if hasattr(self, 'local_tokenizer'):
                del self.local_tokenizer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return a simple error message
            return "I encountered an error while generating a response. Please try a different question or try again later."

    def generate_code(self, query: str, language: str, contexts: List[str], max_context_length: int = 3000) -> str:
        """
        Specialized method for generating code with a more focused prompt
        
        Args:
            query: The user's code request
            language: The programming language to generate code in
            contexts: List of relevant context texts
            max_context_length: Maximum length for the context
            
        Returns:
            Generated code as a string
        """
        print(f"[DEBUG] Generating code for query: {query[:50]}... in {language}")
        try:
            # Join contexts and truncate if too long
            context = "\n\n".join(contexts)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            # Build a specialized prompt for code generation
            prompt = f"""You are a code generator, given the task, generate the code.
            Instructions: 
            - The Code MUST be runnable, complete and without errors. 
            - If Code to be generated is in C or C++ language, then it MUST have a main function 
            - The Code should have all the dependencies, libraries required to compile, build and run the code without error.
            - The Code should not have any SENSITIVE INFORMATION such as passwords or cryptokeys.
            - Should be suitable for deployment rather than prototyping.
            - Include all required header files for C code such as: #include <limits.h>,
              #include <stdarg.h> , #include  <stdio.h>, 
              #include  <stdlib.h>, #include <string.h>, #include <cstring>, #include <unistd.h>, #include <cstdio>
            - Make sure in C if you use malloc type cast it.
            - Make sure in Python don't use Debug mode. For example in FLASK.

             Make sure the code is:\n"
                    - Functional (it should run without errors)\n"
                    - Secure (avoid common vulnerabilities such as input injection, unsafe file handling, etc.)\n"
            
            Generate {language} code for the following request. 
            
Request: {query}

Relevant context:
{context}

Instructions:
1. Provide ONLY the code, with no explanations before or after
2. Use markdown code blocks with the language specified
3. Make sure the code is complete and functional
4. Follow best practices for {language}

```{language}
"""
            
            print(f"[DEBUG] Code generation prompt created, length: {len(prompt)}")
            
            # Generate the response
            response = self.generate_response(prompt)
            
            # Ensure the response ends with a code block closing
            if not response.endswith("```"):
                response += "\n```"
            
            print(f"[DEBUG] Code generation response received, length: {len(response)}")
            return response
        except Exception as e:
            print(f"[ERROR] Error in generate_code: {e}")
            traceback.print_exc()
            return f"Error generating code: {str(e)}"



















