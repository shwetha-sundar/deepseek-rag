import streamlit as st
import main as main

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

st.title("AI Assistant for PDFs 📄")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDFs here!",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    st.rerun()

if st.sidebar.button("Clear DB"):
    try:
        main.clear_database()
        st.success("DB collection cleaned successfully.")
    except Exception as e:
        st.error(f"Error cleaning context: {e}")

# # Streamlit UI to trigger displaying embeddings
# if st.sidebar.button("Show Embeddings"):
#     df = main.display_embeddings()
#     st.dataframe(df)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if uploaded_file:
    saved_files = main.upload_pdfs(uploaded_file)
    db = main.create_vector_store(saved_files)
    st.sidebar.success("Ready to chat!")

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            related_documents = main.retrieve_docs(prompt)
            answer = main.generate_answer(prompt, related_documents)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


