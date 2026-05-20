import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import local modules
from pdf_parser import extract_text_from_pdf, chunk_text, extract_topics_with_llm
from vector_store import RetrieverManager
from generation_engine import GenerationEngine
from answer_evaluator import AnswerEvaluator
from history_manager import HistoryManager
from exporter import generate_paper_pdf, generate_evaluation_report_pdf

# Set Page Config
st.set_page_config(
    page_title="AI Question Paper Generator & Evaluator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Premium Aesthetics
def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Outfit or Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* Apply font family globally */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Main Container Styling */
        .main-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .main-header::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        /* Glassmorphism Cards */
        .premium-card {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.2rem;
            transition: all 0.25s ease-in-out;
        }
        
        .premium-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
            border-color: #cbd5e1;
            transform: translateY(-2px);
        }

        .dark-card {
            background-color: #1e293b;
            color: #f8fafc;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.2rem;
        }

        /* Metric Cards */
        .metric-container {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            flex: 1;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.25rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500;
        }

        /* Question Badge Styles */
        .q-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 0.5rem;
        }

        .badge-mcq { background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
        .badge-sa { background-color: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
        .badge-la { background-color: #f3e8ff; color: #7c3aed; border: 1px solid #e9d5ff; }
        .badge-tf { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .badge-cs { background-color: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }

        .badge-easy { background-color: #ecfdf5; color: #065f46; }
        .badge-medium { background-color: #fffbeb; color: #92400e; }
        .badge-hard { background-color: #fef2f2; color: #991b1b; }

        /* Custom buttons styling */
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
        }

        /* Sidebar enhancement */
        .css-1d391tw {
            background-color: #f8fafc;
        }
        
        /* Highlight sections */
        .section-header {
            color: #1e293b;
            font-weight: 700;
            border-left: 4px solid #3b82f6;
            padding-left: 10px;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Run CSS injection
inject_custom_css()

# Clean API Key helper to handle duplicated sk-sk- or gsk-gsk- prefixes
def clean_api_key(key: str) -> str:
    if not key:
        return ""
    key = key.strip()
    if key.startswith("sk-sk-"):
        key = key[3:]
    elif key.startswith("gsk-gsk-"):
        key = key[4:]
    elif key.startswith("gsk_gsk_"):
        key = key[4:]
    return key

# Initialize Managers
history_mgr = HistoryManager()

# Sidebar Setup
with st.sidebar:
    st.markdown("### Configuration")
    
    # 1. Select Retriever Engine
    retriever_type = st.selectbox(
        "Retrieval Engine",
        options=["FAISS Semantic Search (OpenAI Key)", "Local BM25 Search (Free / No Key)"]
    )
    
    # 2. Select LLM Provider
    ai_provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "Groq"]
    )
    
    openai_key_val = ""
    groq_key_val = ""
    
    # Determine which keys are required
    needs_openai = (retriever_type == "FAISS Semantic Search (OpenAI Key)") or (ai_provider == "OpenAI")
    needs_groq = (ai_provider == "Groq")
    
    # Render OpenAI key input if needed
    if needs_openai:
        openai_env = clean_api_key(os.environ.get("OPENAI_API_KEY", ""))
        if openai_env:
            st.success("OpenAI API Key detected from environment.")
            use_env_openai = st.toggle("Use detected OpenAI Env key", value=True)
            if use_env_openai:
                openai_key_val = openai_env
            else:
                openai_key_val = clean_api_key(st.text_input("Enter OpenAI API Key", type="password"))
        else:
            openai_key_val = clean_api_key(st.text_input("Enter OpenAI API Key", type="password"))
            if not openai_key_val:
                st.warning("OpenAI API Key is required.")
                
    # Render Groq key input if needed
    if needs_groq:
        groq_env = clean_api_key(os.environ.get("GROQ_API_KEY", ""))
        if groq_env:
            st.success("Groq API Key detected from environment.")
            use_env_groq = st.toggle("Use detected Groq Env key", value=True)
            if use_env_groq:
                groq_key_val = groq_env
            else:
                groq_key_val = clean_api_key(st.text_input("Enter Groq API Key", type="password"))
        else:
            groq_key_val = clean_api_key(st.text_input("Enter Groq API Key", type="password"))
            if not groq_key_val:
                st.warning("Groq API Key is required.")

    # 3. Model Selector
    if ai_provider == "OpenAI":
        selected_model = st.selectbox("Model Name", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        selected_model = st.selectbox(
            "Model Name", 
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        )
        
    st.markdown("---")
    
    # Main Navigation
    navigation = st.radio(
        "Navigation",
        options=[
            "Question Paper Generator", 
            "Student Answer Evaluator", 
            "Dashboard & Analytics",
            "Paper History Archives"
        ]
    )
    
    st.markdown("---")
    st.markdown("### About System")
    st.info(
        "This AI-powered assistant uses LLM + RAG (Retrieval-Augmented Generation) "
        "to ingest educational PDFs, generate syllabus-compliant exams, "
        "and perform semantic grading on student answers."
    )
    st.markdown("<small>Powered by Groq / GPT & FAISS / BM25</small>", unsafe_allow_html=True)

# Check for required API Keys
if needs_openai and not openai_key_val:
    st.info("Please enter or confirm your OpenAI API Key in the sidebar to get started.")
    st.stop()
if needs_groq and not groq_key_val:
    st.info("Please enter or confirm your Groq API Key in the sidebar to get started.")
    st.stop()

# Helper function to get Vector DB, Generator and Evaluator
@st.cache_resource
def get_retriever_mgr(key: Optional[str], engine_type: str) -> RetrieverManager:
    engine_val = "faiss" if "FAISS" in engine_type else "bm25"
    return RetrieverManager(api_key=key, engine_type=engine_val)

@st.cache_resource
def get_generation_engine(key: str, base_url: Optional[str]) -> GenerationEngine:
    return GenerationEngine(api_key=key, base_url=base_url)

@st.cache_resource
def get_evaluator(key: str, base_url: Optional[str]) -> AnswerEvaluator:
    return AnswerEvaluator(api_key=key, base_url=base_url)

retriever_key = openai_key_val if "FAISS" in retriever_type else None
retriever_mgr = get_retriever_mgr(retriever_key, retriever_type)

llm_key = openai_key_val if ai_provider == "OpenAI" else groq_key_val
base_url = None if ai_provider == "OpenAI" else "https://api.groq.com/openai/v1"

gen_engine = get_generation_engine(llm_key, base_url)
evaluator = get_evaluator(llm_key, base_url)

# Global helper to print formatted badges
def print_q_badge(q_type: str):
    badges = {
        "MCQ": '<span class="q-badge badge-mcq">MCQ</span>',
        "Short Answer": '<span class="q-badge badge-sa">Short Answer</span>',
        "Long Answer": '<span class="q-badge badge-la">Long Answer</span>',
        "True/False": '<span class="q-badge badge-tf">True/False</span>',
        "Case Study": '<span class="q-badge badge-cs">Case Study</span>',
    }
    return badges.get(q_type, '<span class="q-badge badge-sa">Question</span>')

# ==========================================
# PAGE 1: QUESTION PAPER GENERATOR
# ==========================================
if navigation == "Question Paper Generator":
    # Main Brand Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em;">AI Question Paper Generator</h1>
        <p style="margin:5px 0 0 0; opacity:0.85; font-size:1.05rem; font-weight:300;">
            Upload study resources, customize specifications, and generate curriculum-aligned exams with RAG.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_setup, col_preview = st.columns([1, 1.2])

    with col_setup:
        st.markdown('<h3 class="section-header">1. Upload Resource PDF</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload educational materials (PDF only)", type=["pdf"])
        
        # Session state to store parsing results
        if "pdf_chunks" not in st.session_state:
            st.session_state.pdf_chunks = []
        if "extracted_topics" not in st.session_state:
            st.session_state.extracted_topics = []
        if "pdf_filename" not in st.session_state:
            st.session_state.pdf_filename = ""
            
        if uploaded_file is not None and st.session_state.pdf_filename != uploaded_file.name:
            with st.spinner("⏳ Extracting text and parsing pages..."):
                try:
                    pages = extract_text_from_pdf(uploaded_file)
                    chunks = chunk_text(pages)
                    st.session_state.pdf_chunks = chunks
                    st.session_state.pdf_filename = uploaded_file.name
                    
                    # Store vector store index
                    retriever_mgr.build_index(chunks)
                    
                    # Extract sample text for topic analysis
                    sample_text = "\n".join([p["text"] for p in pages[:3]])
                    extracted_topics = extract_topics_with_llm(
                        text_sample=sample_text,
                        openai_api_key=llm_key,
                        base_url=base_url,
                        model_name=selected_model
                    )
                    st.session_state.extracted_topics = extracted_topics
                    
                    st.success(f"Successfully ingested '{uploaded_file.name}'! Extracted {len(pages)} pages and created {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    st.stop()

        # Display PDF statistics if uploaded
        if st.session_state.pdf_chunks:
            total_words = sum(c["word_count"] for c in st.session_state.pdf_chunks)
            st.markdown(f"""
            <div class="premium-card" style="padding:0.8rem; background-color:#f8fafc;">
                <b>Ingested Document Stats:</b><br/>
                • Total Chunks: {len(st.session_state.pdf_chunks)}<br/>
                • Approx. Word Count: {total_words:,} words<br/>
                • Retrieval Engine: Indexed & Ready ({retriever_type})
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<h3 class="section-header">2. Paper Settings</h3>', unsafe_allow_html=True)
        
        subject = st.text_input("Subject / Course Name", value="Physics 101" if not st.session_state.pdf_filename else st.session_state.pdf_filename.replace(".pdf", ""))
        
        # Populate topics from LLM extraction or fallbacks
        topic_options = st.session_state.extracted_topics if st.session_state.extracted_topics else ["Chapter 1", "Chapter 2", "Chapter 3"]
        selected_topics = st.multiselect(
            "Target Topics / Chapters",
            options=topic_options,
            default=topic_options[:3] if topic_options else []
        )
        
        # If user wants to add a custom topic
        custom_topic = st.text_input("Add Custom Topic (Optional, press Enter)")
        if custom_topic and custom_topic not in selected_topics:
            selected_topics.append(custom_topic)
            st.info(f"Added custom topic: {custom_topic}")

        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
        with col2:
            bloom_levels = st.multiselect(
                "Bloom's Cognitive Level",
                options=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                default=["Understand", "Apply"]
            )
            
        st.markdown("**Question Distribution & Marks**")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            mcq_count = st.number_input("MCQ Qs (1 Mark)", min_value=0, max_value=20, value=3)
        with col_q2:
            sa_count = st.number_input("Short Qs (3 Marks)", min_value=0, max_value=20, value=2)
        with col_q3:
            la_count = st.number_input("Long Qs (5 Marks)", min_value=0, max_value=10, value=1)
            
        custom_instructions = st.text_area("Custom Instructions / Focus Areas", placeholder="e.g. Include at least one numerical problem on momentum, focus on Chapter 2 definitions.")
        
        generate_btn = st.button("Generate Question Paper", type="primary", use_container_width=True)

    with col_preview:
        st.markdown('<h3 class="section-header">3. Generated Output Preview</h3>', unsafe_allow_html=True)
        
        if generate_btn:
            if not st.session_state.pdf_chunks:
                st.error("Please upload a resource PDF first to provide semantic context for the RAG engine.")
            elif not selected_topics:
                st.error("Please select at least one topic.")
            else:
                with st.spinner(" Performing semantic RAG query & generating questions..."):
                    # Auto-initialize/load the index
                    if not retriever_mgr.load_index():
                        if st.session_state.pdf_chunks:
                            retriever_mgr.build_index(st.session_state.pdf_chunks)
                        else:
                            st.error("Retrieval index is not initialized. Please upload a resource PDF first.")
                            st.stop()
                            
                    # Retrieve context based on selected topics
                    search_query = " ".join(selected_topics) + f" {difficulty}"
                    retrieved_chunks = retriever_mgr.search_by_topics(selected_topics, questions_count=(mcq_count + sa_count + la_count))
                    
                    # Prepare specs
                    specs = []
                    if mcq_count > 0:
                        specs.append({"type": "MCQ", "count": mcq_count, "marks": 1})
                    if sa_count > 0:
                        specs.append({"type": "Short Answer", "count": sa_count, "marks": 3})
                    if la_count > 0:
                        specs.append({"type": "Long Answer", "count": la_count, "marks": 5})
                        
                    # Generate paper
                    paper_data = gen_engine.generate_question_paper(
                        subject=subject,
                        topics=selected_topics,
                        difficulty=difficulty,
                        bloom_levels=bloom_levels,
                        question_specs=specs,
                        retrieved_context=retrieved_chunks,
                        custom_instructions=custom_instructions,
                        model_name=selected_model
                    )
                    
                    if "error" in paper_data:
                        st.error(paper_data["error"])
                    else:
                        # Append sources metadata
                        paper_data["sources"] = retrieved_chunks
                        # Save to history
                        paper_id = history_mgr.save_paper(paper_data)
                        st.session_state.current_paper_id = paper_id
                        st.success("Question Paper Generated and Saved to Archives!")

        # Display paper if available in session
        if "current_paper_id" in st.session_state:
            paper_data = history_mgr.get_paper(st.session_state.current_paper_id)
            if paper_data:
                # Beautiful Header Inside Card
                diff_badge = f'<span class="q-badge badge-{paper_data.get("difficulty").lower()}">{paper_data.get("difficulty")}</span>'
                st.markdown(f"""
                <div class="premium-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1e3a8a;">{paper_data.get("title")}</h4>
                        <div>{diff_badge}</div>
                    </div>
                    <div style="font-size:0.9rem; color:#64748b; margin-top:5px;">
                        Subject: <b>{paper_data.get("subject")}</b> | Total Marks: <b>{paper_data.get("total_marks")} Marks</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabbed display for paper, answers and sources
                tab_paper, tab_answers, tab_sources = st.tabs(["Question Paper", "Answer Key", "RAG Sources"])
                
                with tab_paper:
                    # Render Instructions
                    st.markdown("**Instructions:**")
                    for inst in paper_data.get("instructions", []):
                        st.markdown(f"- *{inst}*")
                    st.markdown("---")
                    
                    # Render Questions
                    for q in paper_data.get("questions", []):
                        badge_html = print_q_badge(q.get("type"))
                        st.markdown(f"""
                        <div class="premium-card" style="background-color:#ffffff; border-left:4px solid #3b82f6;">
                            <div style="display:flex; justify-content:space-between;">
                                <div>
                                    {badge_html} 
                                    <span style="font-size:0.8rem; color:#64748b;">Bloom: {q.get('bloom_level', 'Understand')}</span>
                                </div>
                                <b style="color:#1e293b;">[{q.get('marks')} Marks]</b>
                            </div>
                            <div style="margin-top:10px; font-weight:500; font-size:1.05rem;">
                                Q{q.get('id')}. {q.get('text')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show MCQ options
                        if q.get("type") == "MCQ" and q.get("options"):
                            cols = st.columns(2)
                            for i, opt in enumerate(q.get("options", [])):
                                col_idx = i % 2
                                char = chr(65 + i)
                                cols[col_idx].markdown(f"**({char})** {opt}")
                                
                with tab_answers:
                    for q in paper_data.get("questions", []):
                        st.markdown(f"**Q{q.get('id')}. {q.get('text')}**")
                        st.markdown(f"""
                        <div style="background-color:#f0fdf4; border: 1px solid #bbf7d0; border-radius:8px; padding:10px; margin-bottom:15px;">
                            <span style="color:#16a34a; font-weight:bold;">Expected Answer:</span> {q.get('answer')}<br/><br/>
                            <span style="color:#475569; font-weight:bold;">Concept Explanation:</span> <small>{q.get('explanation')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with tab_sources:
                    st.markdown("##### Chunks retrieved from Vector Database for this generation:")
                    for idx, chunk in enumerate(paper_data.get("sources", [])):
                        st.markdown(f"""
                        <div class="premium-card" style="padding:10px; background-color:#f8fafc; font-size:0.85rem;">
                            <b>Source Chunk #{idx+1}</b> (Relevance Score: {round(chunk.get('score', 0)*100)}%) | Page {chunk.get('metadata', {}).get('page_number', 'N/A')}<br/>
                            <p style="color:#475569; font-style:italic; margin-top:5px;">"...{chunk.get('text')}..."</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # Download Buttons
                st.markdown("### Export Documents")
                col_dl1, col_dl2 = st.columns(2)
                
                # Question paper PDF
                paper_pdf_bytes = generate_paper_pdf(paper_data, include_answers=False)
                col_dl1.download_button(
                    label="Download Question Paper PDF",
                    data=paper_pdf_bytes,
                    file_name=f"{paper_data.get('subject').replace(' ', '_')}_Exam.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Answer Key PDF
                answer_pdf_bytes = generate_paper_pdf(paper_data, include_answers=True)
                col_dl2.download_button(
                    label="Download Answer Key PDF",
                    data=answer_pdf_bytes,
                    file_name=f"{paper_data.get('subject').replace(' ', '_')}_AnswerKey.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("No paper selected. Generate one above or view archives.")

# ==========================================
# PAGE 2: STUDENT ANSWER EVALUATOR
# ==========================================
elif navigation == "Student Answer Evaluator":
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em;">AI Student Answer Evaluator</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; font-weight:300;">
            Evaluate student answer papers against reference criteria. Computes semantic match, awards grades, and provides analytical summaries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    papers = history_mgr.load_papers()
    if not papers:
        st.warning("No generated question papers found. Please go to the 'Question Paper Generator' page to create one first.")
        st.stop()
        
    paper_options = {f"{p.get('title')} ({p.get('subject')} - {p.get('total_marks')}M)": p.get("paper_id") for p in papers}
    selected_paper_label = st.selectbox("Select Question Paper", options=list(paper_options.keys()))
    selected_paper_id = paper_options[selected_paper_label]
    paper_data = history_mgr.get_paper(selected_paper_id)
    
    col_input, col_report = st.columns([1, 1.2])
    
    with col_input:
        st.markdown('<h3 class="section-header">Student Answer Submissions</h3>', unsafe_allow_html=True)
        student_name = st.text_input("Student Name", value="John Doe")
        
        # Load questions to let the user answer them or auto-populate demo
        questions = paper_data.get("questions", [])
        student_answers = {}
        
        # Demo Autofill Buttons
        st.markdown("**Simulate/Autofill Demo Answers:**")
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        autofill_good = col_demo1.button("Autofill Good (A+ grade)")
        autofill_med = col_demo2.button("Autofill Medium (C grade)")
        autofill_poor = col_demo3.button("Autofill Poor (F grade)")
        
        demo_answers = {}
        if autofill_good:
            demo_answers = evaluator.generate_mock_student_answers(paper_data, accuracy="good")
            st.success("Filled high-quality answers!")
        elif autofill_med:
            demo_answers = evaluator.generate_mock_student_answers(paper_data, accuracy="medium")
            st.success("Filled partially correct answers!")
        elif autofill_poor:
            demo_answers = evaluator.generate_mock_student_answers(paper_data, accuracy="poor")
            st.success("Filled incorrect/empty answers!")
            
        # Form layout for submitting answers
        for q in questions:
            q_id = str(q.get("id"))
            q_text = q.get("text")
            q_type = q.get("type")
            q_marks = q.get("marks")
            
            st.markdown(f"**Q{q_id}. {q_text} ({q_marks} Marks)**")
            
            # Form field according to question type
            default_val = demo_answers.get(q_id, "") if demo_answers else ""
            
            if q_type == "MCQ" and q.get("options"):
                options = q.get("options")
                # Pre-fill index if demo is selected
                idx = 0
                if default_val in options:
                    idx = options.index(default_val)
                student_answers[q_id] = st.selectbox(f"Select option for Q{q_id}", options=options, index=idx, key=f"ans_{q_id}")
            elif q_type == "True/False":
                options = ["True", "False"]
                idx = 0
                if default_val in options:
                    idx = options.index(default_val)
                student_answers[q_id] = st.selectbox(f"Select True/False for Q{q_id}", options=options, index=idx, key=f"ans_{q_id}")
            else:
                student_answers[q_id] = st.text_area(f"Write answer for Q{q_id}", value=default_val, key=f"ans_{q_id}", height=100)
                
        grade_btn = st.button("Evaluate Answer Paper", type="primary", use_container_width=True)
        
    with col_report:
        st.markdown('<h3 class="section-header">Evaluation Report Card</h3>', unsafe_allow_html=True)
        
        if grade_btn:
            with st.spinner("Grading student answers using semantic evaluation..."):
                report = evaluator.evaluate_answers(paper_data, student_answers, model_name=selected_model)
                
                # Save to history
                eval_id = history_mgr.save_evaluation(report, selected_paper_id, student_name)
                st.session_state.current_eval_report = report
                st.session_state.current_eval_id = eval_id
                st.success("Evaluation complete and saved to archives!")
                
        if "current_eval_report" in st.session_state:
            rep = st.session_state.current_eval_report
            
            # Overall Score Metrics Card
            percentage = rep.get("percentage")
            grade = rep.get("grade")
            score = rep.get("total_score_awarded")
            max_m = rep.get("total_max_marks")
            
            grade_color = "#16a34a" if grade in ["A+", "A", "B"] else ("#ca8a04" if grade in ["C", "D"] else "#dc2626")
            
            st.markdown(f"""
            <div class="dark-card" style="text-align:center;">
                <h3 style="margin:0; color:#3b82f6;">{rep.get('title')}</h3>
                <p style="margin:5px 0 15px 0; color:#94a3b8;">Student: <b>{student_name}</b> | Subject: <b>{rep.get('subject')}</b></p>
                <div style="display:flex; justify-content:space-around; align-items:center;">
                    <div>
                        <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Score Awarded</span>
                        <h2 style="margin:0; font-size:2.2rem; font-weight:800;">{score} / {max_m}</h2>
                    </div>
                    <div>
                        <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Percentage</span>
                        <h2 style="margin:0; font-size:2.2rem; font-weight:800; color:#3b82f6;">{percentage}%</h2>
                    </div>
                    <div>
                        <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Grade</span>
                        <h2 style="margin:0; font-size:2.5rem; font-weight:900; color:{grade_color};">{grade}</h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Print itemized report card
            st.markdown("#### Question-by-Question Scorecard")
            for item in rep.get("items", []):
                status = item.get("status")
                status_badge = ""
                bg_col = "#ffffff"
                border_col = "#e2e8f0"
                
                if status == "Correct":
                    status_badge = '<span class="q-badge" style="background-color:#dcfce7; color:#15803d; border:1px solid #bbf7d0;">Correct</span>'
                    bg_col = "#f0fdf4"
                    border_col = "#bbf7d0"
                elif status == "Partial":
                    status_badge = '<span class="q-badge" style="background-color:#fef9c3; color:#a16207; border:1px solid #fef08a;">Partial Match</span>'
                    bg_col = "#fefce8"
                    border_col = "#fef08a"
                else:
                    status_badge = '<span class="q-badge" style="background-color:#fee2e2; color:#b91c1c; border:1px solid #fecaca;">Incorrect</span>'
                    bg_col = "#fef2f2"
                    border_col = "#fecaca"
                    
                st.markdown(f"""
                <div class="premium-card" style="background-color:{bg_col}; border:1px solid {border_col}; border-left: 4px solid {border_col};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            {status_badge}
                            <span style="font-size:0.8rem; color:#64748b;"><b>Q{item.get('id')}</b> | Type: {item.get('type')}</span>
                        </div>
                        <b style="color:#1e293b;">{item.get('score_awarded')} / {item.get('max_marks')} Marks</b>
                    </div>
                    <div style="margin-top:10px;">
                        <p style="font-weight:500; margin:0 0 5px 0;">Question: {item.get('text')}</p>
                        <p style="font-size:0.9rem; color:#475569; margin:0 0 8px 0;"><b>Student Answer:</b> <i>"{item.get('student_answer') or '[No Answer]'}"</i></p>
                        <hr style="border:0.5px solid {border_col}; margin: 8px 0;"/>
                        <p style="font-size:0.85rem; color:#0f172a; margin:0;">
                            <b>Evaluator Feedback:</b> {item.get('feedback')}
                        </p>
                        {f'<p style="font-size:0.8rem; color:#b91c1c; margin-top:5px;"><b>Missing Concepts:</b> {", ".join(item.get("key_concepts_missing"))}</p>' if item.get("key_concepts_missing") else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Download PDF Report Button
            st.markdown("### Export Evaluation Report")
            eval_pdf_bytes = generate_evaluation_report_pdf(rep, paper_data)
            st.download_button(
                label="Download Evaluation Report PDF",
                data=eval_pdf_bytes,
                file_name=f"Evaluation_{student_name.replace(' ', '_')}_{rep.get('subject').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("Submit student answers on the left panel to trigger grading.")

# ==========================================
# PAGE 3: DASHBOARD & ANALYTICS
# ==========================================
elif navigation == "Dashboard & Analytics":
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em;">Teacher Dashboard & Analytics</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; font-weight:300;">
            Overview of teaching outcomes, evaluation histories, content coverage and performance metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    stats = history_mgr.get_dashboard_stats()
    
    if stats["total_papers"] == 0:
        st.info("Archive is empty. Try creating some question papers first to populate stats.")
        st.stop()
        
    # Top Stats Metric Bar
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_papers']}</div>
            <div class="metric-label">Total Papers Generated</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_evaluations']}</div>
            <div class="metric-label">Total Evaluated Answers</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['average_score']}%</div>
            <div class="metric-label">Class Average Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown('<h4 class="section-header">Subject Distribution</h4>', unsafe_allow_html=True)
        subj_dist = stats["subject_distribution"]
        if subj_dist:
            fig = px.pie(
                names=list(subj_dist.keys()),
                values=list(subj_dist.values()),
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distribution data available.")
            
    with col_g2:
        st.markdown('<h4 class="section-header">Difficulty Distribution</h4>', unsafe_allow_html=True)
        diff_dist = stats["difficulty_distribution"]
        if diff_dist:
            fig = px.bar(
                x=list(diff_dist.keys()),
                y=list(diff_dist.values()),
                labels={'x': 'Difficulty', 'y': 'Count'},
                color=list(diff_dist.keys()),
                color_discrete_map={"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"}
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distribution data available.")
            
    st.markdown('<h4 class="section-header">Subject-wise Average Scores (%)</h4>', unsafe_allow_html=True)
    score_by_subject = stats["score_by_subject"]
    if score_by_subject:
        fig = px.bar(
            x=list(score_by_subject.keys()),
            y=list(score_by_subject.values()),
            labels={'x': 'Subject', 'y': 'Average Score (%)'},
            color_discrete_sequence=["#3b82f6"]
        )
        fig.update_layout(yaxis=dict(range=[0, 100]), height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation scores recorded yet. Submit student answers to display performance trends.")
        
    st.markdown('<h4 class="section-header">Historical Papers Ledger</h4>', unsafe_allow_html=True)
    papers_df = stats["papers_df"]
    if not papers_df.empty:
        # Tweak date display
        papers_df["Created At"] = pd.to_datetime(papers_df["Created At"]).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(
            papers_df[["Paper ID", "Title", "Subject", "Difficulty", "Total Marks", "Questions Count", "Created At"]],
            use_container_width=True
        )

# ==========================================
# PAGE 4: PAPER HISTORY ARCHIVES
# ==========================================
elif navigation == "Paper History Archives":
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #374151 0%, #111827 100%);">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em;">Historical Paper Archives</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; font-weight:300;">
            Manage, load, and re-export previous exam materials or grades records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    papers = history_mgr.load_papers()
    if not papers:
        st.info("The archives are currently empty. Go to the 'Question Paper Generator' to create your first paper.")
        st.stop()
        
    for p in papers:
        p_id = p.get("paper_id")
        created_at_dt = datetime.fromisoformat(p.get("created_at"))
        formatted_date = created_at_dt.strftime('%B %d, %Y at %I:%M %p')
        
        with st.container():
            st.markdown(f"""
            <div class="premium-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.15rem; font-weight:600; color:#1e293b;">{p.get('title')}</span>
                    <span class="q-badge badge-{p.get('difficulty').lower()}">{p.get('difficulty')}</span>
                </div>
                <div style="font-size:0.9rem; color:#475569; margin: 5px 0 10px 0;">
                    Subject: <b>{p.get('subject')}</b> | Total Marks: <b>{p.get('total_marks')} M</b> | Questions: <b>{len(p.get('questions', []))}</b> | Generated on: <i>{formatted_date}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actions buttons layout
            col_act1, col_act2, col_act3, col_act4 = st.columns([1, 1, 1, 1])
            
            # Set this paper as active and go to evaluation
            if col_act1.button(f"Grade Answers", key=f"grade_btn_{p_id}", use_container_width=True):
                st.session_state.current_paper_id = p_id
                st.info(f"Loaded paper '{p.get('title')}' for evaluation! Please navigate to the 'Student Answer Evaluator' tab.")
                
            # Download question paper PDF
            paper_pdf = generate_paper_pdf(p, include_answers=False)
            col_act2.download_button(
                label="Question Paper PDF",
                data=paper_pdf,
                file_name=f"{p.get('subject').replace(' ', '_')}_Exam_{p_id}.pdf",
                mime="application/pdf",
                key=f"dl_paper_{p_id}",
                use_container_width=True
            )
            
            # Download answer key PDF
            answer_pdf = generate_paper_pdf(p, include_answers=True)
            col_act3.download_button(
                label="Answer Key PDF",
                data=answer_pdf,
                file_name=f"{p.get('subject').replace(' ', '_')}_AnswerKey_{p_id}.pdf",
                mime="application/pdf",
                key=f"dl_ans_{p_id}",
                use_container_width=True
            )
            
            # Delete paper
            if col_act4.button(f"Delete Record", key=f"del_{p_id}", type="secondary", use_container_width=True):
                history_mgr.delete_paper(p_id)
                st.success(f"Deleted paper ID: {p_id}")
                st.rerun()
                
            st.markdown("<hr style='border:0.5px solid #e2e8f0; margin:15px 0;'/>", unsafe_allow_html=True)
