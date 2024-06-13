from enum import Enum

class Models(Enum):
    # GroQ Models
    LLAMA_3_8b = 'llama3-8b-8192'
    GEMMA_7b = 'gemma-7b-it'

    # Together AI Models
    MISTRAL_7b = 'mistralai/Mistral-7B-Instruct-v0.3'
    
    # NVIDIA Models
    PHI_3_MINI = 'microsoft/phi-3-mini-128k-instruct'
    PHI_3_SMALL = 'phi-3-small-128k-instruct'