"""
Compatibility layer for huggingface_hub and sentence_transformers
"""
import os
import sys
from huggingface_hub import hf_hub_download

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a cached_download function that wraps hf_hub_download
def cached_download(*args, **kwargs):
    """
    Compatibility wrapper for the deprecated cached_download function
    """
    print("[INFO] Using compatibility wrapper for cached_download")
    # Map old parameters to new ones
    if 'cache_dir' in kwargs:
        kwargs['local_dir'] = kwargs.pop('cache_dir')
    if 'force_download' in kwargs:
        kwargs['force_download'] = kwargs.pop('force_download')
    if 'proxies' in kwargs:
        kwargs['proxies'] = kwargs.pop('proxies')
    if 'resume_download' in kwargs:
        kwargs['resume_download'] = kwargs.pop('resume_download')
    if 'user_agent' in kwargs:
        kwargs['user_agent'] = kwargs.pop('user_agent')
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    
    # Handle positional arguments
    if len(args) > 0:
        # First argument is the URL
        url = args[0]
        # Extract repo_id and filename from URL
        parts = url.split('/')
        if 'resolve' in parts:
            resolve_idx = parts.index('resolve')
            repo_id = '/'.join(parts[parts.index('huggingface.co')+1:resolve_idx])
            filename = '/'.join(parts[resolve_idx+2:])
        else:
            # Fallback to a simple split
            repo_id = '/'.join(parts[parts.index('huggingface.co')+1:parts.index('blob')])
            filename = '/'.join(parts[parts.index('blob')+2:])
        
        # Remove any positional args
        args = []
        
        # Add repo_id and filename to kwargs
        kwargs['repo_id'] = repo_id
        kwargs['filename'] = filename
    
    # Call the new function
    return hf_hub_download(*args, **kwargs)

# Monkey patch huggingface_hub
import huggingface_hub
huggingface_hub.cached_download = cached_download