
# Import required libraries
from dotenv import load_dotenv
from itertools import zip_longest
from streamlit_chat import message
import streamlit as st
# from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from peft import PeftModel
# Load environment variables
load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "nvidia/OpenMath-Mistral-7B-v0.1-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
print(eval_tokenizer)
# eval_prompt = "What are different types of messages?"
# model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# ft_model.eval()
# with torch.no_grad():
#     print(eval_tokenizer.decode(ft_model.generate(**model_input,top_p=0, max_new_tokens=155, repetition_penalty=1.15)[0], skip_special_tokens=True))
    
# Set streamlit page configuration
st.set_page_config(page_title="ChatBot Starter")
st.title("ChatBot Starter")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

# Initialize the ChatOpenAI model
# chat = ChatOpenAI(
#     temperature=0.5,
#     model_name="gpt-3.5-turbo"
# )

# from peft import PeftModel

# chat = PeftModel.from_pretrained(base_model, "F:\shanika\mistral-journal-finetune-CONFDATA\checkpoint-300")

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages =""

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages = human_msg
            # zipped_messages.append(HumanMessage(
            #     content=human_msg))  # Add user messages
        # if ai_msg is not None:
        #     zipped_messages.append(
        #         AIMessage(content=ai_msg))  # Add AI messages
    print(zipped_messages)
    return zipped_messages


def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    # ai_response = chat(zipped_messages)
    model_input = eval_tokenizer(zipped_messages, return_tensors="pt").to("cuda")
    print(type(model_input))
    ft_model = PeftModel.from_pretrained(base_model, "F:\shanika\mistral-journal-finetune-CONFDATA\checkpoint-300")
    ft_model.eval()
    with torch.no_grad():
        answer = eval_tokenizer.decode(ft_model.generate(**model_input,top_p=0, max_new_tokens=155, repetition_penalty=1.15)[0], skip_special_tokens=True)
        print(answer)
        return answer
    

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)


if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')


# # # Add credit
# # st.markdown("""
# # ---
# # Made with 🤖 by [Austin Johnson](https://github.com/AustonianAI)""")
