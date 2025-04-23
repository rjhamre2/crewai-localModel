from llama_cpp import Llama
import torch
from crewai import Agent, Task, Crew
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(f'cuda availability {torch.cuda.is_available()}')
'''
llm = Llama.from_pretrained(
	repo_id="TheBloke/Llama-2-7B-GGUF",
	filename="llama-2-7b.Q2_K.gguf",
    n_gpu_layers = -1,
)   
'''
# Initialize Llama-2-7B

model_path_llama2="E:/Rupesh/Learning/AI-LLM/Agents/CrewAITutorial/models/huggingface_hub/TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q2_K.gguf"
llm = Llama(
    model_path=model_path_llama2,  # Download link below
    n_gpu_layers=35,      # Offload to GPU
    n_ctx=2048,           # Context window
    n_threads=8,          # CPU threads
    verbose=True
)

# Generate text
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    temperature=0.7,
    max_tokens=500
)

print(response)