#!/usr/bin/env python3
"""
Gemini API レスポンス検証スクリプト

Gemini APIを直接呼び出して、レスポンス構造を確認する。
Difyプラグインの問題を調査するためのスクリプト。
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types


def load_api_key():
    """Load API key from project root .env file."""
    env_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip()
    
    return os.environ.get("GEMINI_API_KEY")


def test_quick_search(api_key: str) -> dict:
    """Test Quick Search (Google Search grounding) directly."""
    print("\n" + "=" * 70)
    print("Testing Quick Search (Google Search Grounding)")
    print("=" * 70)
    
    client = genai.Client(api_key=api_key)
    model = "gemini-3-flash-preview"
    query = "東京の今日の天気"
    
    print(f"\nModel: {model}")
    print(f"Query: {query}")
    print("-" * 50)
    
    result = {
        "tool": "quick_search",
        "timestamp": datetime.now().isoformat(),
        "input": {"query": query, "model": model},
        "raw_response": None,
        "parsed_response": None,
        "error": None,
    }
    
    try:
        # Configure Google Search tool
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[google_search_tool])
        
        prompt = f"必ず日本語で回答してください。\n\n{query}"
        
        print("\nSending request to Gemini API...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        
        print("Response received!")
        print("-" * 50)
        
        # Extract content
        content = response.text if response.text else ""
        print(f"\n[Content] (length: {len(content)})")
        if len(content) > 500:
            print(content[:500] + "\n... (truncated)")
        else:
            print(content)
        
        # Extract citations
        citations = []
        search_queries = []
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            print(f"\n[Candidate Info]")
            print(f"  - finish_reason: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")
            
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                print(f"\n[Grounding Metadata]")
                print(f"  - Type: {type(metadata)}")
                
                # Extract search queries
                if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                    search_queries = list(metadata.web_search_queries)
                    print(f"  - Search Queries: {search_queries}")
                
                # Extract citations from grounding chunks
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    print(f"  - Grounding Chunks: {len(metadata.grounding_chunks)}")
                    for i, chunk in enumerate(metadata.grounding_chunks):
                        if hasattr(chunk, "web") and chunk.web:
                            citation = {
                                "url": chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                "title": chunk.web.title if hasattr(chunk.web, "title") else None,
                            }
                            citations.append(citation)
                            print(f"    [{i+1}] {citation['title'] or 'No title'}: {citation['url'][:50]}...")
        
        # Build parsed response (what we would send to Dify)
        parsed = {
            "query": query,
            "content": content,
            "citations": citations,
            "search_queries": search_queries,
            "model": model,
        }
        
        result["parsed_response"] = parsed
        result["success"] = True
        
        print(f"\n[Parsed Response Summary]")
        print(f"  - Content length: {len(content)}")
        print(f"  - Citations: {len(citations)}")
        print(f"  - Search queries: {len(search_queries)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["success"] = False
    
    return result


def test_analyze_urls(api_key: str) -> dict:
    """Test URL Analysis (URL Context) directly."""
    print("\n" + "=" * 70)
    print("Testing Analyze URLs (URL Context)")
    print("=" * 70)
    
    client = genai.Client(api_key=api_key)
    model = "gemini-3-flash-preview"
    urls = ["https://www.google.com"]
    query = "このページの概要を教えてください"
    
    print(f"\nModel: {model}")
    print(f"URLs: {urls}")
    print(f"Query: {query}")
    print("-" * 50)
    
    result = {
        "tool": "analyze_urls",
        "timestamp": datetime.now().isoformat(),
        "input": {"urls": urls, "query": query, "model": model},
        "raw_response": None,
        "parsed_response": None,
        "error": None,
    }
    
    try:
        # Configure URL Context tool
        url_context_tool = types.Tool(url_context=types.UrlContext())
        config = types.GenerateContentConfig(tools=[url_context_tool])
        
        url_list = "\n".join(urls)
        prompt = f"必ず日本語で回答してください。\n\n{query}\n\nURLs:\n{url_list}"
        
        print("\nSending request to Gemini API...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        
        print("Response received!")
        print("-" * 50)
        
        # Extract content
        content = response.text if response.text else ""
        print(f"\n[Content] (length: {len(content)})")
        if len(content) > 500:
            print(content[:500] + "\n... (truncated)")
        else:
            print(content)
        
        # Extract URL metadata
        url_metadata = []
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            print(f"\n[Candidate Info]")
            print(f"  - finish_reason: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")
            
            if hasattr(candidate, "url_context_metadata") and candidate.url_context_metadata:
                metadata = candidate.url_context_metadata
                print(f"\n[URL Context Metadata]")
                print(f"  - Type: {type(metadata)}")
                
                if hasattr(metadata, "url_metadata") and metadata.url_metadata:
                    print(f"  - URL Metadata count: {len(metadata.url_metadata)}")
                    for url_meta in metadata.url_metadata:
                        meta_item = {
                            "url": url_meta.retrieved_url if hasattr(url_meta, "retrieved_url") else "",
                            "status": url_meta.url_retrieval_status if hasattr(url_meta, "url_retrieval_status") else "unknown",
                        }
                        url_metadata.append(meta_item)
                        print(f"    - {meta_item['url']}: {meta_item['status']}")
        
        # Build parsed response
        parsed = {
            "urls": urls,
            "query": query,
            "content": content,
            "url_metadata": url_metadata,
            "model": model,
        }
        
        result["parsed_response"] = parsed
        result["success"] = True
        
        print(f"\n[Parsed Response Summary]")
        print(f"  - Content length: {len(content)}")
        print(f"  - URL metadata: {len(url_metadata)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["success"] = False
    
    return result


def test_search_and_analyze(api_key: str) -> dict:
    """Test Search and Analyze (Google Search + URL Context) directly."""
    print("\n" + "=" * 70)
    print("Testing Search and Analyze (Google Search + URL Context)")
    print("=" * 70)
    
    client = genai.Client(api_key=api_key)
    model = "gemini-3-flash-preview"
    query = "最新のAI技術トレンド"
    
    print(f"\nModel: {model}")
    print(f"Query: {query}")
    print("-" * 50)
    
    result = {
        "tool": "search_and_analyze",
        "timestamp": datetime.now().isoformat(),
        "input": {"query": query, "model": model},
        "raw_response": None,
        "parsed_response": None,
        "error": None,
    }
    
    try:
        # Configure both tools
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(url_context=types.UrlContext()),
        ]
        config = types.GenerateContentConfig(tools=tools)
        
        prompt = f"必ず日本語で回答してください。\n\n{query}"
        
        print("\nSending request to Gemini API...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        
        print("Response received!")
        print("-" * 50)
        
        # Extract content
        content = response.text if response.text else ""
        print(f"\n[Content] (length: {len(content)})")
        if len(content) > 500:
            print(content[:500] + "\n... (truncated)")
        else:
            print(content)
        
        # Extract citations
        citations = []
        search_queries = []
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            print(f"\n[Candidate Info]")
            print(f"  - finish_reason: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")
            
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                print(f"\n[Grounding Metadata]")
                
                if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                    search_queries = list(metadata.web_search_queries)
                    print(f"  - Search Queries: {search_queries}")
                
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    print(f"  - Grounding Chunks: {len(metadata.grounding_chunks)}")
                    for i, chunk in enumerate(metadata.grounding_chunks[:5]):  # Show first 5
                        if hasattr(chunk, "web") and chunk.web:
                            citation = {
                                "url": chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                "title": chunk.web.title if hasattr(chunk.web, "title") else None,
                            }
                            citations.append(citation)
                            print(f"    [{i+1}] {citation['title'] or 'No title'}")
        
        # Build parsed response
        parsed = {
            "query": query,
            "content": content,
            "citations": citations,
            "search_queries": search_queries,
            "model": model,
        }
        
        result["parsed_response"] = parsed
        result["success"] = True
        
        print(f"\n[Parsed Response Summary]")
        print(f"  - Content length: {len(content)}")
        print(f"  - Citations: {len(citations)}")
        print(f"  - Search queries: {len(search_queries)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["success"] = False
    
    return result


def run_test(test_name: str):
    """Run a specific test by name."""
    print("=" * 70)
    print("Gemini API Response Verification")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("\nERROR: GEMINI_API_KEY not found")
        print("Please set it in .env file or environment variable")
        sys.exit(1)
    
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Run specified test
    results = []
    
    if test_name == "quick_search":
        results.append(test_quick_search(api_key))
    elif test_name == "analyze_urls":
        results.append(test_analyze_urls(api_key))
    elif test_name == "search_and_analyze":
        results.append(test_search_and_analyze(api_key))
    elif test_name == "all":
        results.append(test_quick_search(api_key))
        results.append(test_analyze_urls(api_key))
        results.append(test_search_and_analyze(api_key))
    else:
        print(f"Unknown test: {test_name}")
        print("Available tests: quick_search, analyze_urls, search_and_analyze, all")
        sys.exit(1)
    
    # Save results to file
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n\n{'=' * 70}")
    print("Test Summary")
    print("=" * 70)
    
    for result in results:
        status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
        print(f"  {result['tool']}: {status}")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
    else:
        test_name = "quick_search"  # Default to quick_search
    
    run_test(test_name)
