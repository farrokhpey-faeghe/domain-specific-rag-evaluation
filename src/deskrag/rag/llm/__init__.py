"""WS3 (feat/generation): the Ollama client and its on-disk response cache.

Planned: ollama_client.py implements core.interfaces.LLMClient (format=<JSON schema>, temperature 0,
fixed seed, thinking off; asserts the served model matches the requested one; retries twice, then
raises InfraError). cache.py keys responses by sha256(model digest, messages, schema, options).
"""
