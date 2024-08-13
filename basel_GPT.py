import pandas as pd
import numpy as np
import tiktoken
import openai
from scipy import spatial
import streamlit as st

a_1 = "sk-proj-XcGYy0FCItfSapl8Ea-HQxpfaPP8x2vI2z9s1HvPQ1w22UXDoqMZ3iL0Lb4VEtzR6yEcv0"
a_2 = "ZKSTT3BlbkFJd0qVwIxqcz7C_q4eqtT3sKpYQbMr14K5xkR6zoOBUIYzMowQ7cfnUz_aPTH8CJPKqEctKLnJkA"

df = pd.concat([
    pd.read_csv("bs_fw_0.csv"),
    pd.read_csv("bs_fw_1.csv"),
    pd.read_csv("bs_fw_2.csv")    
])


client = openai.OpenAI(api_key=a_1 + a_2)

def num_tokens(text: str, model: str = 'gpt-4o-mini-2024-07-18') -> int:
    """
    Count number of token form text based on OpenAI embedding model
    Input:
        - text: text to be counted
        - model: OpenAI embedding model
    Output:
        - number of token form text
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def string_to_list(text):
    text = text.replace('[', '').replace(']', '').replace(' ', '')
    list_ = [float(x) for x in text.split(',')]
    return list_

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 50,
    cut_off: float = 0.1
) -> tuple[list[str], list[float]]:
    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    """
    query_embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = pd.DataFrame([
        (row["text"], relatedness_fn(query_embedding, string_to_list((row["embedding"]) if isinstance(row["embedding"], str) == True else row["embedding"])))
        for i, row in df.iterrows()
    ])
    strings_and_relatednesses = strings_and_relatednesses.sort_values(1, ascending=False)
    strings_and_relatednesses = strings_and_relatednesses[strings_and_relatednesses[1] >= cut_off]
    strings = strings_and_relatednesses[0]
    relatednesses = strings_and_relatednesses[1]
    return strings[:top_n], relatednesses[:top_n]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """
    Return a message for GPT, with relevant source texts pulled from a dataframe.
    """
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = '''
    Only use the given information below to answer the subsequent question. If cannot answer, do not mention given information.
    '''
    
    question = f"\n\nQuestion: {query}"
    message = introduction + "\n\nInformation:"
    for string in strings:
        if (num_tokens(message +  string + question, model=model) > token_budget):
            break
        else:
            message += "\n\n" + string
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = 'gpt-4o-mini-2024-07-18',
    token_budget: int = 2500 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Basel Framework."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message




st.title("💬 Ask Basel")
st.caption("🚀 A RAG chatbot on [Basel Framework](https://www.bis.org/basel_framework/) powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ask(prompt, df=df)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)





