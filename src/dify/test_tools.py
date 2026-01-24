#!/usr/bin/env python3
"""
Dify プラグインツールのローカルテストスクリプト

使用方法:
    1. 環境変数を設定:
       export GEMINI_API_KEY="your-api-key"
    
    2. テストを実行:
       cd src/dify
       python test_tools.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Any


# モッククラス（Difyプラグインランタイムをシミュレート）
@dataclass
class MockRuntime:
    """Mock runtime for testing tools outside Dify."""
    credentials: dict[str, Any]


class MockTool:
    """Base mock tool class that simulates Dify's Tool class."""
    
    def __init__(self, credentials: dict[str, Any]):
        self.runtime = MockRuntime(credentials=credentials)
    
    def create_json_message(self, data: dict) -> dict:
        """Create a JSON message."""
        return {"type": "json", "data": data}
    
    def create_text_message(self, text: str) -> dict:
        """Create a text message."""
        return {"type": "text", "text": text}


def test_quick_search():
    """Test the QuickSearchTool."""
    print("\n" + "=" * 60)
    print("Testing QuickSearchTool")
    print("=" * 60)
    
    # Import the tool
    from tools.quick_search import QuickSearchTool
    
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        return False
    
    # Create a mock tool instance
    # Note: We need to patch the Tool class to use our mock
    class TestableQuickSearchTool(QuickSearchTool):
        def __init__(self, credentials: dict[str, Any]):
            self.runtime = MockRuntime(credentials=credentials)
    
    tool = TestableQuickSearchTool(credentials={
        "gemini_api_key": api_key,
        "default_quick_model": "gemini-3-flash-preview",
    })
    
    # Test parameters
    params = {
        "query": "今日の東京の天気",
        "language": "ja",
    }
    
    print(f"\nQuery: {params['query']}")
    print("-" * 40)
    
    try:
        for message in tool._invoke(params):
            if message["type"] == "json":
                print("\n[JSON Output]")
                import json
                print(json.dumps(message["data"], indent=2, ensure_ascii=False))
            elif message["type"] == "text":
                print("\n[Text Output]")
                print(message["text"])
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_urls():
    """Test the AnalyzeUrlsTool."""
    print("\n" + "=" * 60)
    print("Testing AnalyzeUrlsTool")
    print("=" * 60)
    
    from tools.analyze_urls import AnalyzeUrlsTool
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        return False
    
    class TestableAnalyzeUrlsTool(AnalyzeUrlsTool):
        def __init__(self, credentials: dict[str, Any]):
            self.runtime = MockRuntime(credentials=credentials)
    
    tool = TestableAnalyzeUrlsTool(credentials={
        "gemini_api_key": api_key,
        "default_quick_model": "gemini-3-flash-preview",
    })
    
    params = {
        "urls": "https://www.google.com",
        "query": "このページの概要を教えてください",
        "language": "ja",
    }
    
    print(f"\nURLs: {params['urls']}")
    print(f"Query: {params['query']}")
    print("-" * 40)
    
    try:
        for message in tool._invoke(params):
            if message["type"] == "json":
                print("\n[JSON Output]")
                import json
                print(json.dumps(message["data"], indent=2, ensure_ascii=False))
            elif message["type"] == "text":
                print("\n[Text Output]")
                print(message["text"])
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_and_analyze():
    """Test the SearchAndAnalyzeTool."""
    print("\n" + "=" * 60)
    print("Testing SearchAndAnalyzeTool")
    print("=" * 60)
    
    from tools.search_and_analyze import SearchAndAnalyzeTool
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        return False
    
    class TestableSearchAndAnalyzeTool(SearchAndAnalyzeTool):
        def __init__(self, credentials: dict[str, Any]):
            self.runtime = MockRuntime(credentials=credentials)
    
    tool = TestableSearchAndAnalyzeTool(credentials={
        "gemini_api_key": api_key,
        "default_quick_model": "gemini-3-flash-preview",
    })
    
    params = {
        "query": "最新のAI技術トレンド",
        "language": "ja",
    }
    
    print(f"\nQuery: {params['query']}")
    print("-" * 40)
    
    try:
        for message in tool._invoke(params):
            if message["type"] == "json":
                print("\n[JSON Output]")
                import json
                print(json.dumps(message["data"], indent=2, ensure_ascii=False))
            elif message["type"] == "text":
                print("\n[Text Output]")
                print(message["text"])
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dify Plugin Tools Test Suite")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("\nERROR: Please set the GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    results = {}
    
    # Run tests
    print("\nSelect a test to run:")
    print("1. Quick Search")
    print("2. Analyze URLs")
    print("3. Search and Analyze")
    print("4. All tests")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-4): ").strip()
    
    if choice == "0":
        print("Exiting...")
        return
    elif choice == "1":
        results["quick_search"] = test_quick_search()
    elif choice == "2":
        results["analyze_urls"] = test_analyze_urls()
    elif choice == "3":
        results["search_and_analyze"] = test_search_and_analyze()
    elif choice == "4":
        results["quick_search"] = test_quick_search()
        results["analyze_urls"] = test_analyze_urls()
        results["search_and_analyze"] = test_search_and_analyze()
    else:
        print("Invalid choice")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")


if __name__ == "__main__":
    main()
