import os
import re
import shutil
import subprocess
import streamlit as st


def init_session_state():
    """Initializes the default session state variables.
    
    This function sets up the initial state for the Streamlit session. It defines a dictionary of default values for various session state variables and then initializes these variables in the Streamlit session state if they don't already exist.
    
    Args:
        None
    
    Returns:
        None
    
    Notes:
        - The function uses a predefined dictionary `SESSION_DEFAULTS` to set the initial values.
        - Session state variables initialized include:
            - messages: An empty list, likely for storing conversation messages.
            - chroma_db: Set to None, possibly for a database connection.
            - db_loaded: A boolean flag, initially set to False.
            - repo_path: A string path to a cloned repository.
            - git_form: A boolean flag, initially set to False.
            - qa: Set to None, possibly for a question-answering system.
            - db_name: Set to None, likely for storing a database name.
        - The function checks if each key from SESSION_DEFAULTS already exists in st.session_state before setting it, ensuring it doesn't overwrite existing values.
    """
    SESSION_DEFAULTS = {
        "messages": [],
        "chroma_db": None,
        "db_loaded": False,
        "repo_path": './cloned_repo',
        "git_form": False,
        "qa": None,
        "db_name": None
    }

    for keys, values in SESSION_DEFAULTS.items():
        if keys not in st.session_state:
            st.session_state[keys] = values


def url_name(url):
    """Extracts a formatted name from a GitHub repository URL.
    
    Args:
        url (str): The GitHub repository URL to process.
    
    Returns:
        str: A formatted string combining the repository owner and name, separated by an underscore.
    
    Raises:
        SystemExit: If the provided URL is not a valid GitHub repository URL.
    """
    pattern = r"https?://github.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        return f"{owner}_{repo}"
    else:
        st.error("Enter valid GitHub URL")
        st.stop()


def clone_repo(git_url, repo_path):
    """Clones a Git repository to a specified local path and removes the .git directory.
    
    Args:
        git_url (str): The URL of the Git repository to clone.
        repo_path (str): The local path where the repository should be cloned.
    
    Returns:
        None
    
    Raises:
        subprocess.CalledProcessError: If the git clone command fails.
    """
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    git_url = git_url.replace(".git", "")

    command = f'git clone {git_url}.git {repo_path} && rm -rf {repo_path}/.git'
    subprocess.run(command, shell=True)


def prompt_format(system_prompt, instruction):
    """Formats a prompt by combining a system prompt and an instruction into a specific template.
    
    Args:
        system_prompt (str): The system prompt to be included in the formatted prompt.
        instruction (str): The instruction to be included in the formatted prompt.
    
    Returns:
        str: A formatted prompt string that combines the system prompt and instruction
             using predefined tags and delimiters.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def model_prompt():
    """Generates a prompt for an AI model using a system prompt and an instruction template.
    
    Args:
        None
    
    Returns:
        str: A formatted prompt string combining the system prompt and instruction template.
    
    Raises:
        None
    
    Notes:
        - The method uses a predefined system prompt and instruction template.
        - The system prompt describes the AI assistant's role and capabilities.
        - The instruction template includes placeholders for context and user questions.
        - The method relies on an external 'prompt_format' function to combine the prompts.
    """
    system_prompt = """You are a helpful assistant, you have good knowledge in coding and you will use the provided context to answer user questions with detailed explanations.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on the provided context, inform the user. Do not use any other information for answering user"""
    instruction = """
    Context: {context}
    User: {question}"""
    return prompt_format(system_prompt, instruction)


def custom_que_prompt():
    """Generate a custom prompt for question rephrasing.
    
    Args:
        None
    
    Returns:
        str: A formatted prompt string containing a system prompt and an instruction prompt for question rephrasing.
    
    """
    que_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question and give only the standalone question as output in the tags <question> and </question>.
    """

    instr_prompt = """Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    return prompt_format(que_system_prompt, instr_prompt)