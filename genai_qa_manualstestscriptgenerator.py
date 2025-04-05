import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Set API key from env or secrets.toml
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("OpenAI API key is missing. Set it in environment or .streamlit/secrets.toml")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt template for test cases
test_case_prompt = PromptTemplate(
    input_variables=["feature_description", "num_test_cases"],
    template="""
You are a QA engineer. Based on the feature description below, generate {num_test_cases} well-structured manual test cases.
Use this format:

1. **Test Case Title**
   - Step: ...
   - Expected: ...

Feature: {feature_description}
"""
)

test_case_chain = LLMChain(llm=llm, prompt=test_case_prompt)

# Streamlit UI
st.set_page_config(page_title="GenAI QA Console", layout="wide")
st.title("GenAI-Powered QA Console")

st.sidebar.header("🔧 Configuration")
num_cases = st.sidebar.slider("Number of Test Cases", 1, 10, 5)

feature_input = st.text_area("Enter Feature Description:", height=150)

if st.button("Generate Test Cases"):
    if feature_input.strip() == "":
        st.warning("Please enter a feature description.")
    else:
        with st.spinner("Thinking like a QA engineer..."):
            try:
                response = test_case_chain.invoke({
                    "feature_description": feature_input,
                    "num_test_cases": num_cases
                })
                st.subheader("Generated Test Cases")
                st.markdown(response["text"])
            except Exception as e:
                st.error(f"Error during generation: {e}")
