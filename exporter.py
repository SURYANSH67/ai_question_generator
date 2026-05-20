import io
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.pdfgen import canvas

# Page Numbering Canvas Helper
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Only on page 2 and later)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AI Generated Question Paper")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 50, 558, 50)
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 35, page_text)
        self.drawString(54, 35, "Confidential - For Academic Use Only")
        self.restoreState()


def generate_paper_pdf(paper_data: Dict[str, Any], include_answers: bool = False) -> bytes:
    """
    Generates a professionally structured PDF for the question paper.
    If include_answers is True, appends the answer key in a separate section.
    """
    buffer = io.BytesIO()
    
    # 0.75-inch margins (54 points)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1e293b")  # Dark Slate Blue
    secondary_color = colors.HexColor("#3b82f6") # Bright Indigo
    text_color = colors.HexColor("#0f172a") # Off Black
    light_bg = colors.HexColor("#f8fafc") # Slate Grey 50
    
    # Typography Styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'PaperSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'QuestionText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=text_color,
        spaceAfter=8
    )
    
    option_style = ParagraphStyle(
        'OptionText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    
    instruction_style = ParagraphStyle(
        'Instructions',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=4
    )
    
    answer_title_style = ParagraphStyle(
        'AnswerTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    story = []
    
    # 1. Header Information Block
    title = paper_data.get("title", "Question Paper").upper()
    story.append(Paragraph(title, title_style))
    
    # Info Table
    subject = paper_data.get("subject", "N/A")
    difficulty = paper_data.get("difficulty", "Medium")
    total_marks = paper_data.get("total_marks", 0)
    
    info_data = [
        [
            Paragraph(f"<b>Subject:</b> {subject}", subtitle_style),
            Paragraph(f"<b>Difficulty:</b> {difficulty}", subtitle_style),
            Paragraph(f"<b>Max Marks:</b> {total_marks}", subtitle_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[168, 168, 168])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    # 2. General Instructions Block
    instructions = paper_data.get("instructions", [])
    if instructions:
        story.append(Paragraph("<b>GENERAL INSTRUCTIONS:</b>", section_heading))
        for inst in instructions:
            story.append(Paragraph(f"• {inst}", instruction_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=15))
    
    # Group questions by type for a beautiful structured layout
    questions = paper_data.get("questions", [])
    mcq_questions = [q for q in questions if q.get("type") == "MCQ"]
    sa_questions = [q for q in questions if q.get("type") in ["Short Answer", "True/False"]]
    la_questions = [q for q in questions if q.get("type") in ["Long Answer", "Case Study"]]
    
    def render_question_list(q_list, section_title):
        if not q_list:
            return
        story.append(Paragraph(section_title, section_heading))
        story.append(Spacer(1, 5))
        
        for q in q_list:
            q_id = q.get("id", "")
            q_text = q.get("text", "")
            q_marks = q.get("marks", 1)
            q_type = q.get("type")
            
            # Question Text + Marks aligned in table to prevent overlaps
            q_para = Paragraph(f"<b>Q{q_id}.</b> {q_text}", body_style)
            marks_para = Paragraph(f"<b>[{q_marks} M]</b>", ParagraphStyle('Marks', parent=body_style, alignment=2, fontName='Helvetica-Bold'))
            
            q_table = Table([[q_para, marks_para]], colWidths=[430, 74])
            q_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(q_table)
            
            # MCQs options layout
            if q_type == "MCQ" and q.get("options"):
                opts = q.get("options", [])
                opt_chars = ["A", "B", "C", "D"]
                # 2x2 table for options
                opt_paras = []
                for i, opt in enumerate(opts):
                    char = opt_chars[i] if i < len(opt_chars) else f"({i+1})"
                    opt_paras.append(Paragraph(f"({char}) {opt}", option_style))
                
                # Zero pad to ensure at least 4 items for MCQ 2x2 grid
                while len(opt_paras) < 4:
                    opt_paras.append(Paragraph("", option_style))
                    
                opt_data = [
                    [opt_paras[0], opt_paras[1]],
                    [opt_paras[2], opt_paras[3]]
                ]
                opt_table = Table(opt_data, colWidths=[252, 252])
                opt_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 15),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ]))
                story.append(opt_table)
                story.append(Spacer(1, 4))
                
            story.append(Spacer(1, 5))
            
    # Render paper sections
    render_question_list(mcq_questions, "SECTION A: MULTIPLE CHOICE QUESTIONS")
    render_question_list(sa_questions, "SECTION B: SHORT ANSWER QUESTIONS")
    render_question_list(la_questions, "SECTION C: LONG ANSWER / ANALYTICAL QUESTIONS")
    
    # 3. Render Answer Key (Optional, on a new page)
    if include_answers:
        story.append(PageBreak())
        
        ans_section_title_style = ParagraphStyle(
            'AnswerSectionTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=15,
            leading=18,
            textColor=secondary_color,
            alignment=1,
            spaceAfter=15,
            keepWithNext=True
        )
        story.append(Paragraph("DETAILED ANSWER KEY & CONCEPT EXPLANATIONS", ans_section_title_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3b82f6"), spaceAfter=15))
        
        for q in questions:
            q_id = q.get("id", "")
            q_text = q.get("text", "")
            correct_ans = q.get("answer", "")
            explanation = q.get("explanation", "")
            q_type = q.get("type", "")
            
            story.append(Paragraph(f"<b>Q{q_id}.</b> {q_text}", body_style))
            
            # Answer block in a clean container
            ans_text = f"<b>Correct Answer:</b> {correct_ans}"
            if q_type == "MCQ" and q.get("options"):
                ans_text = f"<b>Correct Option:</b> {correct_ans}"
                
            ans_para = Paragraph(ans_text, ParagraphStyle('AnsKeyText', parent=body_style, textColor=colors.HexColor("#16a34a"), fontName='Helvetica-Bold'))
            expl_para = Paragraph(f"<b>Pedagogical Explanation:</b> {explanation}", ParagraphStyle('ExplText', parent=body_style, fontSize=9.5, leading=14, textColor=colors.HexColor("#475569")))
            
            ans_container_data = [[ans_para], [expl_para]]
            ans_table = Table(ans_container_data, colWidths=[504])
            ans_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")), # light green
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bbf7d0")),
                ('PADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,1), (-1,1), 4),
            ]))
            story.append(ans_table)
            story.append(Spacer(1, 10))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def generate_evaluation_report_pdf(eval_report: Dict[str, Any], paper_data: Dict[str, Any]) -> bytes:
    """
    Generates a professional evaluation report for a student's answer sheet.
    Includes score breakdowns, semantic matches, and targeted qualitative feedback.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#1e293b")
    secondary_color = colors.HexColor("#0284c7")
    accent_green = colors.HexColor("#16a34a")
    accent_red = colors.HexColor("#dc2626")
    light_bg = colors.HexColor("#f8fafc")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'EvalTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=1,
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'EvalSectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'EvalBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b")
    )
    
    subtext_style = ParagraphStyle(
        'EvalSubtext',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569")
    )
    
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        textColor=primary_color,
        spaceAfter=2
    )

    story = []
    
    # Title
    story.append(Paragraph("STUDENT ANSWER EVALUATION REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=secondary_color, spaceAfter=15))
    
    # Header Info (Subject, Grade, Student, Percent)
    student_name = eval_report.get("student_name", "Student")
    subject = eval_report.get("subject", "N/A")
    total_score = eval_report.get("total_score_awarded", 0)
    max_marks = eval_report.get("total_max_marks", 0)
    percentage = eval_report.get("percentage", 0)
    grade = eval_report.get("grade", "F")
    difficulty = eval_report.get("difficulty", "Medium")
    
    info_data = [
        [
            Paragraph(f"<b>Student:</b> {student_name}", body_style),
            Paragraph(f"<b>Subject:</b> {subject}", body_style),
            Paragraph(f"<b>Paper Difficulty:</b> {difficulty}", body_style)
        ],
        [
            Paragraph(f"<b>Score Awarded:</b> {total_score} / {max_marks}", body_style),
            Paragraph(f"<b>Percentage:</b> {percentage}%", body_style),
            Paragraph(f"<b>Final Grade:</b> <font color='{accent_green.hexval()}'><b>{grade}</b></font>", body_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[168, 168, 168])
    info_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor("#e2e8f0")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("DETAILED EVALUATION BREAKDOWN", section_heading))
    
    # List each question evaluation
    items = eval_report.get("items", [])
    for item in items:
        q_id = item.get("id")
        q_text = item.get("text", "")
        max_m = item.get("max_marks", 1)
        score = item.get("score_awarded", 0.0)
        feedback = item.get("feedback", "")
        status = item.get("status", "Partial")
        student_ans = item.get("student_answer", "")
        similarity = item.get("semantic_similarity", 0.0)
        concepts_cov = item.get("key_concepts_covered", [])
        concepts_mis = item.get("key_concepts_missing", [])
        
        status_color = accent_green if status == "Correct" else (colors.HexColor("#ca8a04") if status == "Partial" else accent_red)
        
        # Draw question card header
        card_header_data = [
            [
                Paragraph(f"<b>Q{q_id}.</b> {q_text[:120]}...", card_title_style),
                Paragraph(f"<b>Score: {score} / {max_m} M</b>", ParagraphStyle('ScoreP', parent=card_title_style, alignment=2, textColor=status_color))
            ]
        ]
        card_header_table = Table(card_header_data, colWidths=[384, 120])
        card_header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        
        # Details inside card
        cov_text = ", ".join(concepts_cov) if concepts_cov else "None"
        mis_text = ", ".join(concepts_mis) if concepts_mis else "None"
        
        details_story = []
        details_story.append(Paragraph(f"<b>Student Answer:</b> <i>\"{student_ans if student_ans else '[No Answer Provided]'}\"</i>", subtext_style))
        details_story.append(Spacer(1, 3))
        details_story.append(Paragraph(f"<b>Concept Coverage Match:</b> {round(similarity * 100)}% | <b>Concepts Covered:</b> {cov_text} | <b>Concepts Missing:</b> {mis_text}", subtext_style))
        details_story.append(Spacer(1, 3))
        details_story.append(Paragraph(f"<b>Assessor Feedback:</b> {feedback}", subtext_style))
        
        # Embed the details in a subtable
        details_table = Table([[details_story]], colWidths=[496])
        details_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        
        # Main outer card table
        outer_card_data = [[card_header_table], [details_table]]
        outer_card_table = Table(outer_card_data, colWidths=[504])
        
        # Color based on status
        bg_card_color = colors.HexColor("#f0fdf4") if status == "Correct" else (colors.HexColor("#fefce8") if status == "Partial" else colors.HexColor("#fef2f2"))
        border_card_color = colors.HexColor("#bbf7d0") if status == "Correct" else (colors.HexColor("#fef08a") if status == "Partial" else colors.HexColor("#fecaca"))
        
        outer_card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_card_color),
            ('BOX', (0,0), (-1,-1), 0.75, border_card_color),
            ('PADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,-1), (-1,-1), 6),
        ]))
        
        story.append(outer_card_table)
        story.append(Spacer(1, 10))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
