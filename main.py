import streamlit as st
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.set_page_config(
    page_title="HackAI 2025",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("HackAI 2025 Project")

tab1, tab2, tab3 = st.tabs(["Upload File", "Key Information", "Chat with LLM"])

with tab1:
    st.header("Upload your PDF file")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key='pdf')
    
    if ss.pdf:
        ss.pdf_ref = ss.pdf
    
    if ss.pdf_ref:
        binary_data = ss.pdf_ref.getvalue()
        pdf_viewer(input=binary_data, width=700, height=1000, key="pdf_viewer")

with tab2:
    if ss.pdf:
        st.header("Here is the key information extracted from the PDF")
        st.write("This section will display the key information extracted from the PDF file.")
    else:
        st.warning("Please upload a PDF file to extract key information.")

with tab3:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there! How can I assist you today?"}
        ]
    
    
    with st.container():
        # Display all messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_prompt = st.chat_input()

        if user_prompt is not None:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            # Process the user prompt and generate a response
            output = "This is a placeholder response."  # Replace with actual LLM processing
            st.session_state.messages.append({"role": "assistant", "content": output})

            st.rerun()