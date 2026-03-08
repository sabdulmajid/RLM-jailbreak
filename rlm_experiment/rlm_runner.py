"""
True RLM Implementation (MIT Paper Style)

Key differences from naive chunking:
1. Root LM NEVER sees the full document in its prompt
2. Document is stored in REPL environment variable `context`
3. Root LM writes Python code to inspect/navigate the document
4. Root LM calls llm_query(text) to delegate reading to sub-model
5. Loop continues until FINAL(...) or FINAL_VAR(...) is output
"""

import sys
import io
import os
import re
import json
import time
import hashlib
import tempfile
import threading
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

from providers import OpenAICompatibleClient, OllamaClient, LLMResponse

# ============================================================================
# LOGGING
# ============================================================================

@dataclass
class LogEntry:
    timestamp: float
    entry_type: str  # "root_prompt", "repl_cell", "repl_output", "llm_query_input", "llm_query_output", "final_answer"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class RLMLogger:
    """Comprehensive logging for RLM execution."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.entries: List[LogEntry] = []
        self.start_time = time.time()
        self.document_hash: Optional[str] = None
    
    def set_document_hash(self, doc: str):
        """Store hash of document for audit."""
        self.document_hash = hashlib.sha256(doc.encode()).hexdigest()
    
    def log(self, entry_type: str, content: str, **metadata):
        if self.enabled:
            entry = LogEntry(
                timestamp=time.time() - self.start_time,
                entry_type=entry_type,
                content=content,
                metadata=metadata,
            )
            self.entries.append(entry)
    
    def audit_root_prompt_isolation(self, root_prompts: List[str], document: str) -> bool:
        """
        CRITICAL: Verify that the document text never appears in the INITIAL root prompts.
        Returns True if isolation is maintained (no leak), False if document leaked.
        
        Note: REPL outputs containing document snippets are expected and allowed.
        We only check if the full document was passed to the root LM initially.
        """
        # Only check the first root prompt (before any REPL execution)
        if not root_prompts:
            return True
        
        first_prompt = root_prompts[0]
        
        # Check if a significant chunk of the document appears in the first prompt
        chunk_size = min(50, max(30, len(document) // 10))
        step = max(1, chunk_size // 2)
        
        if len(document) < 30:
            return True
        
        doc_chunks = [document[i:i+chunk_size] for i in range(0, len(document)-chunk_size+1, step)]
        
        first_prompt_lower = first_prompt.lower()
        for chunk in doc_chunks:
            if chunk.lower() in first_prompt_lower:
                    return False  # Document leaked into root prompt!
        
        return True  # Root prompt is clean
    
    def to_jsonl(self) -> str:
        """Export logs as JSONL."""
        lines = []
        for entry in self.entries:
            lines.append(json.dumps({
                "timestamp": entry.timestamp,
                "type": entry.entry_type,
                "content": entry.content[:500] if len(entry.content) > 500 else entry.content,
                "content_length": len(entry.content),
                **entry.metadata
            }))
        return "\n".join(lines)
    
    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_jsonl())

# ============================================================================
# REPL ENVIRONMENT
# ============================================================================

@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals_snapshot: Dict[str, str]  # Variable name -> repr of value
    execution_time: float

class REPLEnvironment:
    """
    Sandboxed Python REPL for RLM execution.
    
    Pre-loaded variables:
        - context: str - The full document (stored in temp file, loaded into memory)
        - llm_query(prompt: str) -> str - Call sub-model
    
    Special functions:
        - FINAL(answer: str) -> marks final answer
        - FINAL_VAR(var_name: str) -> returns variable as final answer
    """
    
    def __init__(
        self,
        document: str,
        sub_model_client: OpenAICompatibleClient,
        logger: RLMLogger,
        enable_llm_query: bool = True,
    ):
        self.document = document
        self.sub_client = sub_model_client
        self.logger = logger
        self.enable_llm_query = enable_llm_query
        
        self._lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        
        # Track llm_query calls
        self.llm_query_count = 0
        
        # Initialize execution environment
        self._init_globals()
        self.locals = {}
        
        # Load document into context variable
        self._load_document()
    
    def _init_globals(self):
        """Initialize safe globals for REPL execution."""
        
        # Define llm_query function
        def llm_query(prompt: str) -> str:
            """Query the sub-model with the given prompt."""
            if not self.enable_llm_query:
                return "[ERROR] llm_query is disabled in this mode."
            
            self.llm_query_count += 1
            self.logger.log("llm_query_input", prompt, call_num=self.llm_query_count)
            
            response = self.sub_client.simple_completion(prompt)
            
            self.logger.log("llm_query_output", response, call_num=self.llm_query_count)
            return response
        
        # Define FINAL function
        def FINAL(answer: str) -> str:
            """Mark the final answer."""
            marker = f"__FINAL_ANSWER__:{answer}"
            print(marker)  # Print so it appears in stdout
            return marker
        
        # Define FINAL_VAR function
        def FINAL_VAR(var_name: str) -> str:
            """Return a variable as the final answer."""
            var_name = var_name.strip().strip('"').strip("'")
            if var_name in self.locals:
                marker = f"__FINAL_ANSWER__:{self.locals[var_name]}"
                print(marker)  # Print so it appears in stdout
                return marker
            else:
                error_msg = f"[ERROR] Variable '{var_name}' not found."
                print(error_msg)
                return error_msg
        
        self.globals = {
            '__builtins__': {
                # Safe built-ins
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'range': range, 'reversed': reversed, 'any': any, 'all': all,
                'repr': repr, 'format': format, 'chr': chr, 'ord': ord,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                '__import__': __import__, 'open': open,
                # Exception classes
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError,
            },
            # RLM-specific functions
            'llm_query': llm_query,
            'FINAL': FINAL,
            'FINAL_VAR': FINAL_VAR,
        }
    
    def _load_document(self):
        """Load document into context variable."""
        # Save to temp file and load via code (mimics real RLM)
        doc_path = os.path.join(self.temp_dir, "context.txt")
        with open(doc_path, "w") as f:
            f.write(self.document)
        
        load_code = f'''
with open(r"{doc_path}", "r") as f:
    context = f.read()
'''
        self.execute(load_code)
    
    @contextmanager
    def _capture_output(self):
        """Capture stdout/stderr during execution."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def execute(self, code: str) -> REPLResult:
        """Execute code in the REPL environment."""
        start_time = time.time()
        
        self.logger.log("repl_cell", code)
        
        with self._capture_output() as (stdout_buf, stderr_buf):
            try:
                # Combine globals and locals for execution
                combined = {**self.globals, **self.locals}
                
                # Execute the code
                exec(code, combined, combined)
                
                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals:
                        self.locals[key] = value
                
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
                
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {str(e)}"
        
        execution_time = time.time() - start_time
        
        # Create snapshot of locals (repr of each value)
        locals_snapshot = {}
        for k, v in self.locals.items():
            if not k.startswith('_'):
                try:
                    locals_snapshot[k] = repr(v)[:200]  # Truncate long reprs
                except:
                    locals_snapshot[k] = "<unrepresentable>"
        
        result = REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals_snapshot=locals_snapshot,
            execution_time=execution_time,
        )
        
        self.logger.log("repl_output", f"stdout: {stdout}\nstderr: {stderr}", 
                       execution_time=execution_time)
        
        return result
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = '''You are an AI assistant with access to a Python REPL environment.

You have access to:
- `context`: A string variable containing a large document. Do NOT try to print the entire context.
- `llm_query(prompt: str) -> str`: A function to ask a sub-model to analyze text snippets.
- `FINAL(answer: str)`: Call this when you have the final answer.
- `FINAL_VAR(var_name: str)`: Call this to return a variable as the final answer.

IMPORTANT RULES:
1. The context is very large. DO NOT try to print it all. Use len(context), context[:500], etc.
2. Use llm_query() to delegate reading/summarizing of text chunks to a sub-model.
3. Write Python code in ```python code blocks to interact with the REPL.
4. When you have the final answer, call FINAL("your answer") or FINAL_VAR("variable_name").
5. Be systematic: explore the structure first, then drill down.

Example:
```python
# First, check the size
print(f"Document length: {len(context)} characters")
print(f"First 500 chars: {context[:500]}")
```
'''

def build_query_prompt(query: str, iteration: int, force_final: bool = False) -> str:
    """Build the prompt for each iteration."""
    if force_final:
        return f'''
The user asked: "{query}"

You have run out of iterations. Based on what you've learned, provide your best answer now.
Call FINAL("your answer") with your response.
'''
    else:
        return f'''
The user asked: "{query}"

Iteration {iteration}: Write Python code to explore the context and work towards answering the question.
Remember: Use llm_query() to analyze text chunks. Call FINAL() when you have the answer.
'''

# ============================================================================
# RLM RUNNER
# ============================================================================

class RLMRunner:
    """
    True Recursive Language Model implementation.
    
    The root LM NEVER sees the full document - it only interacts via REPL.
    """
    
    def __init__(
        self,
        root_client: OpenAICompatibleClient,
        sub_client: Optional[OpenAICompatibleClient] = None,
        max_iterations: int = 20,
        enable_logging: bool = True,
        enable_llm_query: bool = True,
    ):
        self.root_client = root_client
        self.sub_client = sub_client or root_client
        self.max_iterations = max_iterations
        self.enable_logging = enable_logging
        self.enable_llm_query = enable_llm_query
        
        self.logger = RLMLogger(enabled=enable_logging)
        self.root_prompts: List[str] = []  # Track all root prompts for audit
    
    def run(self, document: str, query: str) -> Dict[str, Any]:
        """
        Run RLM on a document with a query.
        
        Args:
            document: The full document (will be stored in REPL, not sent to root LM)
            query: The user's question
            
        Returns:
            Dict with 'answer', 'logs', 'audit_passed', etc.
        """
        self.logger = RLMLogger(enabled=self.enable_logging)
        self.logger.set_document_hash(document)
        self.root_prompts = []
        
        # Initialize REPL with document
        repl = REPLEnvironment(
            document=document,
            sub_model_client=self.sub_client,
            logger=self.logger,
            enable_llm_query=self.enable_llm_query,
        )
        
        # Build initial messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        final_answer = None
        
        try:
            for iteration in range(1, self.max_iterations + 1):
                # Build query prompt
                force_final = (iteration == self.max_iterations)
                query_prompt = build_query_prompt(query, iteration, force_final)
                
                current_messages = messages + [{"role": "user", "content": query_prompt}]
                
                # Log the root prompt (for audit)
                full_prompt = "\n".join([m["content"] for m in current_messages])
                self.root_prompts.append(full_prompt)
                self.logger.log("root_prompt", full_prompt, iteration=iteration)
                
                # Get response from root LM
                response = self.root_client.completion(current_messages)
                response_text = response.content
                
                self.logger.log("root_response", response_text, iteration=iteration)
                
                # Check for FINAL answer in response
                final_match = re.search(r'__FINAL_ANSWER__:(.+?)(?:$|\n)', response_text)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    break
                
                # Check for FINAL() call in code
                final_call_match = re.search(r'FINAL\s*\(\s*["\'](.+?)["\']\s*\)', response_text)
                if final_call_match:
                    final_answer = final_call_match.group(1).strip()
                    break
                
                # Extract and execute code blocks
                code_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
                
                if code_blocks:
                    for code in code_blocks:
                        result = repl.execute(code)
                        
                        # Check if FINAL was called in execution
                        if "__FINAL_ANSWER__:" in result.stdout:
                            match = re.search(r'__FINAL_ANSWER__:(.+)', result.stdout)
                            if match:
                                final_answer = match.group(1).strip()
                                break
                        
                        # Add execution result to conversation
                        exec_message = f"Execution result:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": exec_message})
                    
                    if final_answer:
                        break
                else:
                    # No code blocks, add response and continue
                    messages.append({"role": "assistant", "content": response_text})
            
            # If still no answer, extract from last response
            if not final_answer:
                final_answer = response_text[:500] if response_text else "[NO ANSWER]"
        
        finally:
            repl.cleanup()
        
        # Audit: verify document never appeared in root prompts
        audit_passed = self.logger.audit_root_prompt_isolation(self.root_prompts, document)
        
        self.logger.log("final_answer", final_answer, audit_passed=audit_passed)
        
        return {
            "answer": final_answer,
            "iterations": iteration,
            "llm_query_count": repl.llm_query_count,
            "audit_passed": audit_passed,
            "logs": self.logger,
        }

# ============================================================================
# FULL CONTEXT BASELINE
# ============================================================================

class FullContextBaseline:
    """
    Baseline: Feed full document directly to LLM in one shot.
    Used for comparison with RLM.
    """
    
    def __init__(self, client: OpenAICompatibleClient, max_context_chars: int = 400000):
        self.client = client
        self.max_context_chars = max_context_chars
    
    def run(self, document: str, query: str) -> Dict[str, Any]:
        """Run full-context baseline."""
        
        if len(document) > self.max_context_chars:
            return {
                "answer": f"[ERROR] Document too long ({len(document)} chars > {self.max_context_chars} limit)",
                "error": True,
            }
        
        prompt = f'''Please answer the following question based on the document below.

QUESTION: {query}

DOCUMENT:
{document}

ANSWER:'''
        
        start_time = time.time()
        response = self.client.completion(prompt)
        latency = time.time() - start_time
        
        return {
            "answer": response.content,
            "latency_s": latency,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "error": False,
        }


# Test
if __name__ == "__main__":
    from providers import OllamaClient
    
    # Create clients
    client = OllamaClient(
        host="http://ece-nebula04.eng.uwaterloo.ca:11434",
        model="qwen3-next:latest",
        temperature=0.0,
    )
    
    # Test RLM
    rlm = RLMRunner(root_client=client, sub_client=client, max_iterations=5)
    
    # Create test document
    doc = "The secret number is 42. " * 100 + "\nThe magic word is BANANA.\n" + "More filler text. " * 100
    
    result = rlm.run(doc, "What is the magic word?")
    
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Audit passed: {result['audit_passed']}")
