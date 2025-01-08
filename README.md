# LangChain Basics

This workspace contains examples of using LangChain with Streamlit to build various AI-driven applications. Each script showcases a different aspect of LangChain, from basic prompt usage to more advanced retrieval and agent functionality.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
```sh
pip install -r requirements.txt
```
3. Set your OpenAI API key as an environment variable:
```sh
export OPENAI_API_KEY="your-api-key"
```

## Usage

To run any of the scripts below, use:
```sh
streamlit run <script_name.py>
```

For example:
```sh
streamlit run 01-basic-prompt.py
```

## Files

- [01-basic-prompt.py](01-basic-prompt.py): Basic prompt handling using [`ChatOpenAI`](01-basic-prompt.py) to generate responses.  
- [02-prompt-template.py](02-prompt-template.py): Demonstrates prompt templates with [`ChatPromptTemplate`](02-prompt-template.py) for structured conversations.  
- [03-retrieval-chain.py](03-retrieval-chain.py): Shows how to fetch information from a loaded dataset using `create_retrieval_chain`.  
- [04-conversation-retreival.py](04-conversation-retreival.py): Builds on retrieval chains by maintaining a conversation history for context-aware Q&A.  
- [05-agents.py](05-agents.py): Implements an agent architecture with various tools and a prompt template for more sophisticated interactions.

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- OpenAI API key

## License

This project is licensed under the MIT License.

## Developed By

Naman Adep
