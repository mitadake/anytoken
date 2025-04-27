import streamlit as st
from transformers import AutoTokenizer
from functools import lru_cache
import random
# from dotenv import load_dotenv
import os

# load_dotenv() 

hf_token = os.getenv("HF_API_KEY")

if hf_token:
    print("HuggingFace token loaded successfully.")
else:
    print("Error: HuggingFace token not found!")


st.set_page_config(page_title="HF Tokenizer Visualizer", page_icon="🔖", layout="wide")

st.title("HF Tokenizer Visualizer 🔖")
st.write("Type a HuggingFace model ID and some text to see how it gets tokenized!")

@st.cache_resource
@lru_cache(maxsize=5)
def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_token  
    )
    return tokenizer

# st.sidebar.header("Settings")
# model_id = st.sidebar.text_input("HuggingFace Model ID", value="gpt2")

st.sidebar.header("Settings")

popular_models = [
    "gpt2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "google/gemma-7b",
    "codellama/CodeLlama-70b-hf",
    "microsoft/phi-4",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-72B"
]

selected_model = st.sidebar.selectbox(
    "Pick a popular model or type your own:",
    options=popular_models + ["Custom input..."]
)

if selected_model == "Custom input...":
    model_id = st.sidebar.text_input("Enter custom HuggingFace model ID", value="gpt2")
else:
    model_id = selected_model


input_text = st.text_area("Input Text", value="The world is full of amazing")

if st.button("Tokenize!"):
    try:
        tokenizer = load_tokenizer(model_id)

        encoded = tokenizer(input_text, add_special_tokens=False, return_tensors=None)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        token_ids = encoded["input_ids"]

        st.success(f"Loaded tokenizer: `{model_id}`")
        
        with st.expander("Tokenizer Info ⚙️"):
            st.write(f"**Model Type:** {tokenizer.__class__.__name__}")
            st.write(f"**Vocab Size:** {tokenizer.vocab_size}")
            
            special_tokens = tokenizer.special_tokens_map
            if special_tokens:
                st.write("**Special Tokens:**")
                st.json(special_tokens)
            else:
                st.write("_No special tokens found._")

        st.subheader("Tokenized Output")

        html_tokens = []

        for token, token_id in zip(tokens, token_ids):
            color = f"hsl({random.randint(0, 360)}, 70%, 80%)"
            span = f'<span style="background-color:{color}; padding:4px; margin:2px; border-radius:5px;">{token}</span>'
            html_tokens.append(span)

        full_html = " ".join(html_tokens)

        st.markdown(full_html, unsafe_allow_html=True)

        with st.expander("Copy Tokens 📋"):
            tokens_str = " ".join(tokens) 
            st.code(tokens_str, language="text")

        with st.expander("See tokens as table"):
            st.table({
                "Token": tokens,
                "Token ID": token_ids
            })

    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")

st.markdown("---")
