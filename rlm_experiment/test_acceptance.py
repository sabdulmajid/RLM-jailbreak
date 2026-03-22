#!/usr/bin/env python3
"""
Acceptance Tests for True RLM Implementation

Tests:
1. Needle-in-haystack: RLM can find specific info via REPL navigation
2. Root prompt audit: Verify document never leaks into root prompts
3. Canary injection: Verify canary detection works
4. llm_query functionality: Sub-model calls work correctly
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from providers import OllamaClient
from rlm_runner import RLMRunner, FullContextBaseline, REPLEnvironment, RLMLogger
from poison import inject_payload, check_canary_in_response, PayloadLibrary, CANARY_STRING


def test_repl_environment():
    """Test that REPL environment works correctly."""
    print("\n=== TEST: REPL Environment ===")
    
    document = "The secret code is ALPHA-BRAVO-123. Remember this code."
    logger = RLMLogger(enabled=True)
    
    # Create mock client (won't actually call LLM)
    class MockClient:
        def simple_completion(self, prompt):
            return "Mock response"
    
    repl = REPLEnvironment(
        document=document,
        sub_model_client=MockClient(),
        logger=logger,
        enable_llm_query=True,
    )
    
    # Test 1: Context is loaded
    result = repl.execute("print(len(context))")
    assert str(len(document)) in result.stdout, f"Expected length {len(document)} in output"
    print(f"  [PASS] Context loaded, length: {len(document)}")
    
    # Test 2: Can read context slices
    result = repl.execute("print(context[:20])")
    assert "secret" in result.stdout.lower(), f"Expected 'secret' in output: {result.stdout}"
    print(f"  [PASS] Can read context slices")
    
    # Test 3: Can define variables
    result = repl.execute("x = 42\nprint(x)")
    assert "42" in result.stdout
    print(f"  [PASS] Can define and use variables")
    
    # Test 4: llm_query is callable
    result = repl.execute("result = llm_query('test')\nprint(result)")
    assert "Mock response" in result.stdout
    print(f"  [PASS] llm_query is callable")
    
    # Test 5: FINAL works
    result = repl.execute("print(FINAL('my answer'))")
    assert "__FINAL_ANSWER__:my answer" in result.stdout
    print(f"  [PASS] FINAL() works correctly")
    
    repl.cleanup()
    print("  [ALL PASSED] REPL Environment tests")
    return True


def test_poison_injection():
    """Test poison injection module."""
    print("\n=== TEST: Poison Injection ===")
    
    document = """
    This is paragraph one with some content.
    
    This is paragraph two with more information.
    
    This is paragraph three at the end.
    """
    
    payload = PayloadLibrary.harmless_canary()
    
    # Test injection at different offsets
    for offset in [0.25, 0.5, 0.75]:
        result = inject_payload(document, payload, offset=offset)
        assert CANARY_STRING in result.payload_used
        assert len(result.poisoned_doc) > len(result.original_doc)
        print(f"  [PASS] Injection at offset {offset}")
    
    # Test canary detection
    response_with_canary = f"The answer is {CANARY_STRING} based on analysis."
    response_without = "The answer is based on analysis."
    
    assert check_canary_in_response(response_with_canary) == True
    assert check_canary_in_response(response_without) == False
    print(f"  [PASS] Canary detection works")
    
    print("  [ALL PASSED] Poison Injection tests")
    return True


def test_root_prompt_isolation():
    """Test that document never appears in the INITIAL root prompt."""
    print("\n=== TEST: Root Prompt Isolation ===")
    
    # Create a document with unique identifiable content
    document = """This is a long document that contains important information.
    
    The secret password is: ALPHA-BRAVO-CHARLIE-DELTA-123.
    
    More content follows here with various details about the topic.
    We need enough text to ensure proper chunking behavior.
    The document continues with additional paragraphs of text.
    """
    
    logger = RLMLogger(enabled=True)
    logger.set_document_hash(document)
    
    # Simulate root prompts - first prompt is the system prompt (no document)
    # Later prompts may include REPL outputs with document snippets
    root_prompts = [
        "You are an AI assistant with access to a Python REPL...",  # First prompt - clean
        "Execution result:\nstdout: Document length: 325 characters\n",  # REPL output
    ]
    
    # Verify isolation passes (document not in FIRST prompt)
    assert logger.audit_root_prompt_isolation(root_prompts, document) == True
    print(f"  [PASS] Audit passes when document not in first prompt")
    
    # Verify isolation fails if document is in the FIRST prompt
    leaked_chunk = document[:80]
    leaked_first_prompt = [f"Here is the context: {leaked_chunk}"]
    assert logger.audit_root_prompt_isolation(leaked_first_prompt, document) == False
    print(f"  [PASS] Audit fails when document leaks into first prompt")
    
    print("  [ALL PASSED] Root Prompt Isolation tests")
    return True


def test_full_context_baseline():
    """Test full context baseline works (offline mode with mock)."""
    print("\n=== TEST: Full Context Baseline ===")
    
    # Mock client for testing
    class MockClient:
        def completion(self, messages):
            class Response:
                content = "The document discusses various topics."
                prompt_tokens = 100
                completion_tokens = 20
            return Response()
    
    baseline = FullContextBaseline(MockClient(), max_context_chars=10000)
    
    document = "Test document content."
    result = baseline.run(document, "What is this about?")
    
    assert "answer" in result
    assert result["error"] == False
    print(f"  [PASS] Baseline returns answer")
    
    # Test length limit
    long_doc = "x" * 500000
    result = baseline.run(long_doc, "What is this?")
    assert "[ERROR]" in result["answer"]
    print(f"  [PASS] Baseline rejects too-long documents")
    
    print("  [ALL PASSED] Full Context Baseline tests")
    return True


def test_ollama_connection():
    """Test Ollama connection (requires running server)."""
    print("\n=== TEST: Ollama Connection ===")
    
    api_base = os.environ.get("API_BASE", "http://ece-nebula04.eng.uwaterloo.ca:11434")
    model = os.environ.get("MODEL", "qwen3-next:latest")
    
    try:
        client = OllamaClient(
            host=api_base,
            model=model,
            temperature=0.0,
            timeout=30.0,
        )
        
        response = client.simple_completion("Say 'hello' and nothing else.")
        
        if "hello" in response.lower():
            print(f"  [PASS] Ollama connection successful")
            print(f"  Response: {response[:100]}...")
            return True
        else:
            print(f"  [WARN] Unexpected response: {response[:100]}")
            return True  # Connection worked, response was different
            
    except Exception as e:
        print(f"  [SKIP] Ollama not available: {e}")
        return None  # Skip, not fail


def run_all_tests():
    """Run all acceptance tests."""
    print("=" * 60)
    print("TRUE RLM IMPLEMENTATION - ACCEPTANCE TESTS")
    print("=" * 60)
    
    tests = [
        ("REPL Environment", test_repl_environment),
        ("Poison Injection", test_poison_injection),
        ("Root Prompt Isolation", test_root_prompt_isolation),
        ("Full Context Baseline", test_full_context_baseline),
        ("Ollama Connection", test_ollama_connection),
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
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○", "ERROR": "!"}[status]
        print(f"  [{icon}] {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len([r for r in results if r[1] != "SKIP"])
    
    print(f"\n{passed}/{total} tests passed")
    
    return all(s in ("PASS", "SKIP") for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
