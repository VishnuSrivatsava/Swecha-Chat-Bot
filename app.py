import streamlit as st
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
import os
from time import sleep

from dotenv import load_dotenv
load_dotenv()

yourHFtoken = os.getenv("YOUR_HF_TOKEN")

av_us = 'images/man.png'
av_ass = 'images/robot.png'

st.title("Swecha Chat Bot")
st.subheader("using Starchat-beta")

if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = "HuggingFaceH4/starchat-beta"

def starchat(model,myprompt, your_template):
    from langchain import PromptTemplate, LLMChain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
    llm = HuggingFaceHub(repo_id=model , 
                         model_kwargs={"min_length":30,
                                       "max_new_tokens":256, "do_sample":True, 
                                       "temperature":0.2, "top_k":50, 
                                       "top_p":0.95, "eos_token_id":49155})
    template = your_template
    prompt = PromptTemplate(template=template, input_variables=["myprompt"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_reply = llm_chain.run(myprompt)
    reply = llm_reply.partition('')[0]
    return reply


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.image(av_us, use_column_width=True, caption="User Avatar")
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.image(av_ass, use_column_width=True, caption="Assistant Avatar")
            st.markdown(message["content"])

if myprompt := st.chat_input("What is an AI model?"):
    st.session_state.messages.append({"role": "user", "content": myprompt})
    with st.chat_message("user"):
        st.image(av_us, use_column_width=True, caption="User Avatar")
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        writehistory(usertext)
    with st.chat_message("assistant"):
        st.image(av_ass, use_column_width=True, caption="Assistant Avatar")
        message_placeholder = st.empty()
        full_response = ""
        res  =  starchat(
                st.session_state["hf_model"],
                myprompt, "\n\n\n{myprompt}\n")
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        writehistory(asstext)       
        st.session_state.messages.append({"role": "assistant", "content": full_response})

