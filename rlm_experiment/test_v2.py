"""
Unit Tests for RLM v2 - Bounded Snippet Access

Tests:
1. llm_query fails when input exceeds MAX_SNIPPET_CHARS
2. read_span enforces MAX_SPAN_CHARS limit
3. search works correctly
4. FINAL detection works
5. Determinism sanity check
"""

import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from rlm_runner_v2 import (
    BoundedREPL, RunLog, MAX_SNIPPET_CHARS, MAX_SPAN_CHARS,
    SnippetTooLargeError, run_baseline, run_rlm
)


def test_llm_query_size_limit():
    """Test that llm_query raises SnippetTooLargeError for large inputs."""
    print("\n=== TEST: llm_query size limit ===")
    
    # Create a document larger than MAX_SNIPPET_CHARS
    large_doc = "x" * (MAX_SNIPPET_CHARS + 1000)
    
    log = RunLog(
        condition="test",
        document_hash="test",
        document_length=len(large_doc),
        query="test",
    )
    
    # Mock client that shouldn't be called
    class MockClient:
        def completion(self, messages):
            raise AssertionError("llm_query should not call client with oversized input")
    
    repl = BoundedREPL(large_doc, MockClient(), "test", log)
    
    # Directly test the llm_query function by accessing it from globals
    llm_query_fn = repl.globals['llm_query']
    
    try:
        # Try to call with oversized input
        llm_query_fn("x" * (MAX_SNIPPET_CHARS + 100))
        print("  [FAIL] Expected SnippetTooLargeError but none was raised")
        return False
    except SnippetTooLargeError as e:
        print(f"  [PASS] Correctly raised SnippetTooLargeError: {str(e)[:80]}...")
        return True


def test_llm_query_with_context():
    """Test that llm_query(context) fails when context > MAX_SNIPPET_CHARS."""
    print("\n=== TEST: llm_query(context) fails ===")
    
    # Create document larger than limit
    document = "The secret is here. " * 500  # ~10000 chars
    assert len(document) > MAX_SNIPPET_CHARS, f"Document too small: {len(document)}"
    
    log = RunLog(
        condition="test",
        document_hash="test",
        document_length=len(document),
        query="test",
    )
    
    class MockClient:
        def completion(self, messages):
            raise AssertionError("Should not be called")
    
    repl = BoundedREPL(document, MockClient(), "test", log)
    
    # Get the llm_query function
    llm_query_fn = repl.globals['llm_query']
    read_span_fn = repl.globals['read_span']
    doc_length_fn = repl.globals['doc_length']
    
    # Get full document via read_span (will be truncated to MAX_SPAN_CHARS)
    # But then try to pass more than allowed to llm_query
    full_text = "x" * (MAX_SNIPPET_CHARS + 100)
    
    try:
        llm_query_fn(full_text)
        print("  [FAIL] Expected SnippetTooLargeError")
        return False
    except SnippetTooLargeError:
        print("  [PASS] llm_query correctly rejects oversized input")
        return True


def test_read_span_limit():
    """Test that read_span enforces MAX_SPAN_CHARS limit."""
    print("\n=== TEST: read_span size limit ===")
    
    document = "x" * 10000
    
    log = RunLog(
        condition="test",
        document_hash="test",
        document_length=len(document),
        query="test",
    )
    
    class MockClient:
        pass
    
    repl = BoundedREPL(document, MockClient(), "test", log)
    
    # Request more than MAX_SPAN_CHARS
    code = f'''
span = read_span(0, {MAX_SPAN_CHARS + 5000})
print(len(span))
'''
    
    stdout, stderr = repl.execute(code)
    
    # Check that result is truncated
    result_len = int(stdout.strip())
    if result_len <= MAX_SPAN_CHARS:
        print(f"  [PASS] read_span truncated to {result_len} chars (limit: {MAX_SPAN_CHARS})")
        return True
    else:
        print(f"  [FAIL] read_span returned {result_len} chars, expected <= {MAX_SPAN_CHARS}")
        return False


def test_search_function():
    """Test that search function works correctly."""
    print("\n=== TEST: search function ===")
    
    document = """
    This is some text. The secret is BANANA.
    More text follows here with various content.
    Another mention of BANANA appears here.
    """
    
    log = RunLog(
        condition="test",
        document_hash="test", 
        document_length=len(document),
        query="test",
    )
    
    class MockClient:
        pass
    
    repl = BoundedREPL(document, MockClient(), "test", log)
    
    code = '''
results = search("BANANA")
print(len(results))
for r in results:
    print(r['start'], r['end'])
'''
    
    stdout, stderr = repl.execute(code)
    lines = stdout.strip().split('\n')
    
    if int(lines[0]) >= 2:
        print(f"  [PASS] search found {lines[0]} matches for 'BANANA'")
        return True
    else:
        print(f"  [FAIL] Expected >= 2 matches, got {lines[0]}")
        return False


def test_final_detection():
    """Test that FINAL() is correctly detected."""
    print("\n=== TEST: FINAL detection ===")
    
    document = "Test document"
    
    log = RunLog(
        condition="test",
        document_hash="test",
        document_length=len(document),
        query="test",
    )
    
    class MockClient:
        pass
    
    repl = BoundedREPL(document, MockClient(), "test", log)
    
    code = '''
FINAL("The answer is 42")
'''
    
    stdout, stderr = repl.execute(code)
    
    if "__FINAL_ANSWER__:The answer is 42" in stdout:
        print("  [PASS] FINAL() correctly outputs marker")
        return True
    else:
        print(f"  [FAIL] Expected FINAL marker in stdout: {stdout}")
        return False


def test_determinism(num_trials: int = 5):
    """Test that seed + temperature=0 gives stable output."""
    print(f"\n=== TEST: Determinism ({num_trials} trials) ===")
    
    try:
        from providers_v2 import get_client
    except ImportError:
        print("  [SKIP] providers_v2 not available")
        return None
    
    try:
        client = get_client(temperature=0.0, seed=42)
    except Exception as e:
        print(f"  [SKIP] Could not create client: {e}")
        return None
    
    prompt = "What is 2 + 2? Reply with just the number."
    responses: List[str] = []
    
    for i in range(num_trials):
        try:
            response = client.simple_completion(prompt, seed=42)
            responses.append(response.strip())
            print(f"  Trial {i+1}: {response[:50]}")
        except Exception as e:
            print(f"  Trial {i+1}: ERROR - {e}")
            responses.append(f"ERROR: {e}")
    
    # Check if all responses are identical
    unique_responses = set(responses)
    if len(unique_responses) == 1 and "ERROR" not in list(unique_responses)[0]:
        print(f"  [PASS] All {num_trials} responses identical - determinism confirmed")
        return True
    else:
        print(f"  [WARN] Got {len(unique_responses)} unique responses - seed may not guarantee determinism")
        print(f"  Unique responses: {unique_responses}")
        return False


def test_no_direct_context_access():
    """Test that 'context' variable is not directly accessible."""
    print("\n=== TEST: No direct context access ===")
    
    document = "Secret document content"
    
    log = RunLog(
        condition="test",
        document_hash="test",
        document_length=len(document),
        query="test",
    )
    
    class MockClient:
        pass
    
    repl = BoundedREPL(document, MockClient(), "test", log)
    
    # Check that 'context' is not in globals or locals
    has_context_in_globals = 'context' in repl.globals
    has_context_in_locals = 'context' in repl.locals
    
    if not has_context_in_globals and not has_context_in_locals:
        print("  [PASS] Direct 'context' access blocked (not in globals/locals)")
        return True
    else:
        print(f"  [FAIL] 'context' accessible: globals={has_context_in_globals}, locals={has_context_in_locals}")
        return False


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("RLM v2 UNIT TESTS - Bounded Snippet Access")
    print("=" * 60)
    print(f"MAX_SNIPPET_CHARS: {MAX_SNIPPET_CHARS}")
    print(f"MAX_SPAN_CHARS: {MAX_SPAN_CHARS}")
    
    tests = [
        ("llm_query size limit", test_llm_query_size_limit),
        ("llm_query(context) fails", test_llm_query_with_context),
        ("read_span size limit", test_read_span_limit),
        ("search function", test_search_function),
        ("FINAL detection", test_final_detection),
        ("No direct context access", test_no_direct_context_access),
        ("Determinism", test_determinism),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is None:
                results.append((name, "SKIP"))
            elif result:
                results.append((name, "PASS"))
            else:
                results.append((name, "FAIL"))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○", "ERROR": "!", "WARN": "?"}[status]
        print(f"  [{icon}] {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len([r for r in results if r[1] not in ("SKIP",)])
    
    print(f"\n{passed}/{total} tests passed")
    
    return all(s in ("PASS", "SKIP", "WARN") for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
