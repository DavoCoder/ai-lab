from ui.labs.app_mode import AppMode
import streamlit as st
from data_processing.documet_processor import DocumentProcessor
from typing import List, Any
import io
import zipfile
import json
from pathlib import Path

class DataProcessing(AppMode):
    def __init__(self):
        self.processor = DocumentProcessor()

    @staticmethod
    def render():
        processor = DocumentProcessor()
        
        st.sidebar.header("Processing Options")
        
        process_type = st.sidebar.selectbox(
            "Process Type",
            ["Text Cleaning", "Document Splitting", "Format Conversion", 
             "Metadata Extraction", "Batch Processing"]
        )
        
        file_types = st.sidebar.multiselect(
            "File Types",
            ["PDF", "TXT", "DOCX", "HTML", "JSON"]
        )

        if not file_types:
            st.warning("Please select at least one file type.")
            return

        uploaded_files = st.file_uploader(
            "Upload Files", 
            accept_multiple_files=True,
            type=[processor.SUPPORTED_EXTENSIONS[ft] for ft in file_types]
        )

        if not uploaded_files:
            st.info("Please upload files to process.")
            return

        # Process files based on selected options
        if process_type == "Text Cleaning":
            DataProcessing._handle_text_cleaning(processor, uploaded_files)
        elif process_type == "Document Splitting":
            DataProcessing._handle_document_splitting(processor, uploaded_files)
        elif process_type == "Format Conversion":
            DataProcessing._handle_format_conversion(processor, uploaded_files)
        elif process_type == "Metadata Extraction":
            DataProcessing._handle_metadata_extraction(processor, uploaded_files)
        elif process_type == "Batch Processing":
            DataProcessing._handle_batch_processing(processor, uploaded_files)

    @staticmethod
    def _handle_text_cleaning(processor: DocumentProcessor, files: List[Any]):
        st.subheader("Text Cleaning Options")
        
        options = {
            "remove_special_chars": st.checkbox("Remove Special Characters"),
            "remove_extra_spaces": st.checkbox("Remove Extra Spaces"),
            "remove_urls": st.checkbox("Remove URLs"),
            "remove_emails": st.checkbox("Remove Email Addresses"),
            "normalize_whitespace": st.checkbox("Normalize Whitespace"),
            "lowercase": st.checkbox("Convert to Lowercase")
        }
        
        if st.button("Clean Text"):
            for file in files:
                try:
                    content = processor.read_file_content(file)
                    cleaned_content = processor.clean_text(content, options)
                    
                    st.download_button(
                        f"Download Cleaned {Path(file.name).stem}",
                        cleaned_content,
                        file_name=f"cleaned_{Path(file.name).stem}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

    @staticmethod
    def _handle_document_splitting(processor: DocumentProcessor, files: List[Any]):
        st.subheader("Document Splitting Options")
        
        split_config = {
            "method": st.selectbox(
                "Split Method",
                ["By Paragraphs", "By Sentences", "By Word Count", "By Custom Delimiter"]
            )
        }
        
        if split_config["method"] == "By Word Count":
            split_config["words_per_chunk"] = st.number_input(
                "Words per chunk", 
                min_value=50, 
                value=500
            )
        elif split_config["method"] == "By Custom Delimiter":
            split_config["delimiter"] = st.text_input("Custom Delimiter")
        
        if st.button("Split Documents"):
            for file in files:
                try:
                    content = processor.read_file_content(file)
                    chunks = processor.split_document(content, split_config)
                    
                    # Create ZIP file with chunks
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        for i, chunk in enumerate(chunks, 1):
                            chunk_filename = f"chunk_{i}_{Path(file.name).stem}.txt"
                            zf.writestr(chunk_filename, chunk)
                    
                    st.download_button(
                        f"Download Split {Path(file.name).stem}",
                        zip_buffer.getvalue(),
                        file_name=f"split_{Path(file.name).stem}.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"Error splitting {file.name}: {str(e)}")

    @staticmethod
    def _handle_format_conversion(processor: DocumentProcessor, files: List[Any]):
        st.subheader("Format Conversion Options")
        
        target_format = st.selectbox(
            "Convert To",
            ["TXT", "JSON", "CSV", "HTML"]
        )
        
        if st.button("Convert Files"):
            for file in files:
                try:
                    content = processor.read_file_content(file)
                    converted_content, output_ext = processor.convert_format(
                        content, 
                        target_format
                    )
                    
                    st.download_button(
                        f"Download Converted {Path(file.name).stem}",
                        converted_content,
                        file_name=f"converted_{Path(file.name).stem}{output_ext}",
                        mime=f"text/{target_format.lower()}"
                    )
                    
                except Exception as e:
                    st.error(f"Error converting {file.name}: {str(e)}")

    @staticmethod
    def _handle_metadata_extraction(processor: DocumentProcessor, files: List[Any]):
        st.subheader("Metadata Extraction")
        
        for file in files:
            try:
                metadata = processor.extract_metadata(file)
                
                # Display metadata in expandable section
                with st.expander(f"Metadata for {file.name}"):
                    st.json(metadata)
                
                # Allow downloading metadata as JSON
                st.download_button(
                    f"Download Metadata for {file.name}",
                    json.dumps(metadata, indent=2),
                    file_name=f"metadata_{file.name}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error extracting metadata from {file.name}: {str(e)}")

    @staticmethod
    def _handle_batch_processing(processor: DocumentProcessor, files: List[Any]):
        st.subheader("Batch Processing Options")
        
        operations = st.multiselect(
            "Select Operations",
            ["Clean Text", "Extract Metadata", "Convert Format"]
        )
        
        if not operations:
            st.warning("Please select at least one operation.")
            return
        
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(files):
                try:
                    result = processor.batch_process(file, operations)
                    results.append(result)
                    
                    # Update progress
                    progress = (idx + 1) / len(files)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if results:
                # Create ZIP file with results
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zf:
                    for result in results:
                        base_name = Path(result['filename']).stem
                        if "clean_text" in result['operations']:
                            zf.writestr(
                                f"cleaned_{base_name}.txt",
                                result['operations']['clean_text']
                            )
                        if "metadata" in result['operations']:
                            zf.writestr(
                                f"metadata_{base_name}.json",
                                json.dumps(result['operations']['metadata'], indent=2)
                            )
                
                st.success(f"Processed {len(results)} files successfully")
                st.download_button(
                    "Download Batch Results",
                    zip_buffer.getvalue(),
                    file_name="batch_results.zip",
                    mime="application/zip"
                )

    @staticmethod
    def _show_error_message(error: str):
        """Helper method to display error messages consistently."""
        st.error(f"Error: {error}")

    @staticmethod
    def _show_success_message(message: str):
        """Helper method to display success messages consistently."""
        st.success(message)