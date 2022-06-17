import streamlit as st
# import tensorflow as tf 
# from   tensorflow import qna
from transformers import pipeline
import streamlit.components.v1 as components


# bootstrap 4 collapse example
components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div class="jumbotron">
      <h1 class="display-4">Q&A APP</h1>
      <p class="lead">Answers based on your paragraph.</p>
      <hr class="my-4">
    </div>
    
    """,
    height=300,
)

@st.cache(allow_output_mutation = True)
def load_module():
     model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
     return model
     
def main():
    qa = load_module()
    st.title("Ask questions based on your article")
    articles = st.text_area("Please enter your article")
    quest = st.text_input("Ask your question based on the article")
    button = st.button("Answer")
    with st.spinner("Finding Answer..."):
        if button and articles:
            answer = qa(question=quest , context=articles)
            #answer = qa.findAnswers(question, passage)
            st.success(answer['answer'])

if __name__=='__main__':
    main()