import streamlit as st
import pdfplumber
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Document


import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize Hugging Face Embedding Model and LLM (Groq LLaMA)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(model="llama3-8b-8192",api_key=GROQ_API_KEY)

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def generate_interview_questions(candidate_name, job_description, resume_text):
    """Generates structured interview questions based on Job Description and Resume."""
    # Convert job description and resume text into Document objects
    job_doc = Document(text=job_description)
    resume_doc = Document(text=resume_text)

    # Create a vector index from the document
    job_index = VectorStoreIndex.from_documents([job_doc], embed_model=embed_model)
    # Build Indexes
    #job_index = VectorStoreIndex.from_documents([job_description], embed_model=embed_model)
    #resume_index = VectorStoreIndex.from_documents([resume_text], embed_model=embed_model)
    resume_index = VectorStoreIndex.from_documents([resume_doc], embed_model=embed_model)

    # Create Query Engines
    job_query_engine = job_index.as_query_engine(llm=llm)
    resume_query_engine = resume_index.as_query_engine(llm=llm)

    # Generate interview questions
    prompt = f"""
    You are a technical interview expert. Your task is to generate interview questions for the candidate, {candidate_name}, based on the provided Job Description and Resume.

    1. From the Job Description:
    - Generate 3 questions that assess {candidate_name}'s fundamental understanding of the key concepts (easy level).
    - Generate 3 questions that require deeper knowledge and experience related to the core responsibilities and technologies mentioned in the Job Description (medium level).
    - Ensure that each question includes context from the Job Description, explaining why it is relevant to the role and how it relates to what the candidate will be expected to do on a day-to-day basis.
    - Prioritize roles, responsibilities, and the key expectations from the candidate as outlined in the Job Description.

    2. From the Resume:
    - Generate 1 question that assesses {candidate_name}'s fundamental understanding based on their listed skills and experiences (easy level).
    - Generate 2 questions that require deeper knowledge and experience, focusing on the candidate's previous work experience, specific technologies they have worked with, projects they have completed, and the skills they have developed (medium level).
    - Ensure that each question references specific skills, experiences, or accomplishments from {candidate_name}'s resume, explaining why it is being asked.

    3. General Guidelines:
    - The questions should flow naturally as part of an ongoing conversation between the interviewer and {candidate_name}.
    - Ensure that the questions are well-structured, relevant, and personalized, focusing on the specific technologies, methodologies, and experiences mentioned in both the Job Description and Resume.
    - The tone and language should be appropriate for a real interview setting, making the interaction feel genuine and engaging.
    - Do not mention question numbering. List the questions without numbering and ensure the format is consistent.

    4. Level of Difficulty:
    - Consider the candidate's years of experience when generating questions. Tailor the difficulty (easy, medium, hard) based on their experience level.
    - Easy questions should generally come from the skills section, while medium and hard questions should delve into the candidate's experience and deeper understanding of the technologies and responsibilities outlined.

    5. Specific Considerations:
    - Ensure more context is provided in the questions derived from the Job Description compared to those from the Resume.
    - Consider the section "roles and responsibilities/what company is expecting from candidate" in the Job Description when generating questions.
    - Assume you are conducting a real interview and thoroughly review both the Job Description and Resume to generate relevant and comprehensive interview questions.
    """

    response = job_query_engine.query(prompt)
    return response

# Streamlit UI
st.title("AI-Powered Interview Question Generator")

# Upload Job Description and Resume PDFs
job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
candidate_name = st.text_input("Enter Candidate's Name")

if job_file and resume_file and candidate_name:
    # Extract text from Job Description and Resume
    job_description = extract_text_from_pdf(job_file)
    resume_text = extract_text_from_pdf(resume_file)
    
    # Show extracted text (optional for debugging)
    st.text_area("Extracted Job Description", job_description, height=150)
    st.text_area("Extracted Resume", resume_text, height=150)

    if st.button("Generate Interview Questions"):
        # Generate Interview Questions
        interview_questions = generate_interview_questions(candidate_name, job_description, resume_text)
        
        # Convert list of questions to formatted string
        questions_text = "\n\n".join(interview_questions) if isinstance(interview_questions, list) else str(interview_questions)

        # Display the generated questions
        st.text_area("Generated Interview Questions", questions_text, height=400)

        # Provide a download option
        st.download_button(
            label="Download Interview Questions",
            data=questions_text.encode("utf-8"),  # Convert string to bytes
            file_name=f"{candidate_name}_Interview_Questions.md",
            mime="text/markdown",
        )
    '''
    if st.button("Generate Interview Questions"):
        # Generate Interview Questions
        interview_questions = generate_interview_questions(candidate_name, job_description, resume_text)
        
        # Display the generated questions
        st.text_area("Generated Interview Questions", interview_questions, height=400)

        st.download_button(
            label="Download Job Description as Markdown",
            data=interview_questions.encode("utf-8"),  # Convert string to bytes
            file_name=f"{job_title}_interviewQuestions.md",
            mime="text/markdown",
        )
'''
