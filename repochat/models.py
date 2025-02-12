from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def hf_embeddings():
    """Creates and returns a HuggingFaceEmbeddings object with a specific pre-trained model.
    
    Args:
        None
    
    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with the 'sentence-transformers/all-mpnet-base-v2' model.
    """
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2",
    )

def code_llama():
    """Initialize and return a LlamaCpp language model instance.
    
    Args:
        None
    
    Returns:
        LlamaCpp: An instance of the LlamaCpp language model configured with specific parameters.
    
    Raises:
        FileNotFoundError: If the specified model file is not found.
        ValueError: If any of the numeric parameters are out of valid range.
    """
    callbackmanager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="D:/Chatlocal/pythonProject/models/codellama-7b.Q3_K_M.gguf",
        n_ctx=2048,
        max_tokens=250,
        n_gpu_layers=10,
        n_batch=512,
        f16_kv=True,
        callback_manager=callbackmanager,
        verbose=True,
        use_mlock=True
    )
    return llm

