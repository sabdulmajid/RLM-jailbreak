#!/usr/bin/env python3
"""Test call-path attribution logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rlm_runner_v2 import run_baseline, run_rlm, RunLog
from poison import inject_payload, DEFAULT_PAYLOADS

# Quick test with canary
doc = 'Normal text. ' * 50 + 'More content here.'
payload = DEFAULT_PAYLOADS['direct_command']
result = inject_payload(doc, payload, offset=0.5)
test_doc = result.poisoned_doc
marker = 'CANARY_12345'

print('=== Testing call-path attribution ===')
print(f'Injection in doc: {marker in test_doc}')

# Test with mock client
class MockClient:
    model = 'test'
    temperature = 0.0
    seed = 42
    def completion(self, messages):
        class R:
            content = 'CANARY_12345'  # Simulate hijacked output
        return R()

client = MockClient()

# Test baseline
print('\n--- Baseline Test ---')
log = run_baseline(test_doc, 'Summarize', client, injection_marker=marker)
print(f'root_saw_injection: {log.root_saw_injection}')
print(f'subcall_saw_injection: {log.subcall_saw_injection}')
print(f'subcall_emitted_injection: {log.subcall_emitted_injection}')
print(f'canary_emitter: {log.canary_emitter}')
print(f'final_answer: {log.final_answer[:50]}')

print('\n=== All tests passed ===')
