import streamlit as st
from doc_processor import extract_text_from_file, classify_and_route_document

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Back Office Document Classifier",
    page_icon="📄",
    layout="centered"
)

# --- App Title and Description ---
st.title("📄 Back Office Document Classification & Routing Agent")
st.write(
    "Upload a document (PDF or TXT) such as an invoice, purchase order, or contract. "
    "The AI agent will classify it and suggest the correct department for routing."
)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a document...",
    type=["pdf", "txt"],
    help="Upload your document here. Supported formats: PDF, TXT."
)

# --- Main Logic ---
if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner(f"Analyzing '{uploaded_file.name}'... This may take a moment for scanned documents."):
        # 1. Extract text from the uploaded file
        document_text = extract_text_from_file(uploaded_file)

        if document_text and not document_text.startswith("Error"):
            # 2. Classify and route the document using the LLM
            result = classify_and_route_document(document_text)

            # 3. Display the results
            st.success("Analysis Complete!")
            st.subheader("Classification Results")

            if "error" in result:
                st.error(f"An error occurred: {result['error']}")
            else:
                # Display results in a structured format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="**Document Type**",
                        value=result.get("classification", "N/A")
                    )
                with col2:
                    st.metric(
                        label="**Confidence Score**",
                        value=f"{result.get('confidence_score', 0)}%"
                    )

                st.info(f"**Recommended Routing:** {result.get('routing_department', 'N/A')}")
                st.write("**Reasoning:**")
                st.write(f"> {result.get('reasoning', 'No reasoning provided.')}")

        else:
            st.error(f"Could not extract text from the document. Details: {document_text}")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by **Streamlit**, **Ollama**, and **Tesseract OCR**.")