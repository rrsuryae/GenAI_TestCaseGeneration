import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import base64

# Initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("Please set your OpenAI API key in environment or .streamlit/secrets.toml")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain setup
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Prompt for generating manual test cases
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

# Prompt for generating automated test scripts
auto_test_script_prompt = PromptTemplate(
    input_variables=["manual_test_cases"],
    template="""
You are a test automation engineer. Convert the following manual test cases into pytest-style automated test scripts using Playwright.

Manual Test Cases:
{manual_test_cases}
"""
)
auto_test_script_chain = LLMChain(llm=llm, prompt=auto_test_script_prompt)

# Streamlit UI
st.set_page_config(page_title="GenAI QA Console", layout="wide")
st.title("GenAI-Powered QA Console")

# Sidebar controls
st.sidebar.header("🔧 Configuration")
num_cases = st.sidebar.slider("Number of Test Cases", 1, 10, 5)

# Main input area
feature_input = st.text_area("Enter Feature Description:", height=150)

# Initialize session state
if "manual_test_cases" not in st.session_state:
    st.session_state.manual_test_cases = ""

# Generate manual test cases
if st.button("Generate Test Cases"):
    if feature_input.strip() == "":
        st.warning("Please enter a feature description.")
    else:
        with st.spinner("Generating test cases..."):
            response = test_case_chain.invoke({
                "feature_description": feature_input,
                "num_test_cases": num_cases
            })
            st.session_state.manual_test_cases = response["text"] if isinstance(response, dict) else response

# Display manual test cases if available
if st.session_state.manual_test_cases:
    st.subheader("Generated Manual Test Cases")
    st.markdown(st.session_state.manual_test_cases)

    # Button to generate automated test script
    if st.button("Generate Automated Test Script"):
        with st.spinner("Generating test automation script..."):
            auto_response = auto_test_script_chain.invoke({
                "manual_test_cases": st.session_state.manual_test_cases
            })
            script_text = auto_response["text"] if isinstance(auto_response, dict) else auto_response
            st.subheader("Generated Pytest Test Script")
            st.code(script_text, language="python")

            # Add download button
            b64 = base64.b64encode(script_text.encode()).decode()
            href = f'<a href="data:file/text;base64,{b64}" download="generated_test_script.py">Download Test Script</a>'
            st.markdown(href, unsafe_allow_html=True)

# Placeholder for future modules
st.divider()
st.subheader("Coming Next: Code Generator & Test Executor")
st.info("Stay tuned for auto-code, testing and self-healing features!")
