import os
import io
import uuid
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
import tempfile


# Load environment variables and initialize clients
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

from openai import OpenAI
client = OpenAI(api_key=api_key)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conf = ConnectionConfig(
    MAIL_USERNAME="api",
    MAIL_PASSWORD="47dd5af7b90e63d1c1ebaf2977fa3deb",
    MAIL_FROM="joemama@demomailtrap.co",
    MAIL_PORT=587,
    MAIL_SERVER="live.smtp.mailtrap.io",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

# -----------------------------
# Helper functions for JSON persistence
# -----------------------------
def load_db(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_db(file_path: str, data: dict):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

# File paths for persistence
PATIENTS_DB_PATH = "patients_db.json"
FILES_DB_PATH = "files_db.json"
REPORTS_DB_PATH = "reports_db.json"
CHAT_DB_PATH = "chat_db.json"

# Load "databases" from JSON files
patients_db: Dict[str, dict] = load_db(PATIENTS_DB_PATH)
files_db: Dict[str, dict] = load_db(FILES_DB_PATH)
reports_db: Dict[str, dict] = load_db(REPORTS_DB_PATH)
chat_db: Dict[str, List[dict]] = load_db(CHAT_DB_PATH)

# -----------------------------
# Pydantic Models
# -----------------------------
class Patient(BaseModel):
    id: str
    name: str
    nhs_number: str

class FileRecord(BaseModel):
    id: str
    patient_id: str
    filename: str
    extracted_text: str

class Report(BaseModel):
    id: str
    patient_id: str
    file_id: str
    complexity: str
    initial_report: Optional[str] = None
    critique_feedback: Optional[str] = None
    final_report: Optional[str] = None

class ChatMessage(BaseModel):
    sender: str  # "doctor" or "patient"
    message: str

# -----------------------------
# GPT API Helper Function (using new structure)
# -----------------------------
async def gpt_call(messages: List[dict]) -> str:
    """
    Call the ChatGPT API using the new OpenAI client.
    """
    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# GPT Agents for Report Generation (Updated Prompts)
# -----------------------------
async def agent_1_generate_report(context_text: str, complexity: str) -> str:
    prompt = (
        "Generate a detailed and personalized medical report for a patient based on the following context. "
        "Remove all markdown symbols (such as asterisks or underscores), avoid duplicate greetings, and do not include any date or greeting information. "
        "Ensure the report includes clear section headings such as SUMMARY, MEDICAL HISTORY, CURRENT SYMPTOMS, DIAGNOSTIC DETAILS, and RECOMMENDED ACTIONS. "
        "Use plain language and an empathetic tone. "
        f"Complexity level: {complexity}. Context:\n{context_text}"
    )
    messages = [
        {"role": "system", "content": "You are a medical expert who generates detailed, patient-friendly reports with clear section headings and plain language. Do not include any greeting or date information in your output."},
        {"role": "user", "content": prompt},
    ]
    return await gpt_call(messages)

async def agent_2_critique(report: str) -> str:
    prompt = (
        "Critique the following detailed medical report. Identify any areas where the language is too technical, "
        "explanations are unclear, or where further detail would help the reader. Provide suggestions to improve clarity and personalization:\n"
        f"{report}"
    )
    messages = [
        {"role": "system", "content": "You are a medical communications expert who ensures reports are clear, empathetic, and patient-friendly."},
        {"role": "user", "content": prompt},
    ]
    return await gpt_call(messages)

async def agent_3_refine(report: str, critique: str) -> str:
    prompt = (
        "Refine the following medical report by addressing these critiques:\n"
        f"{critique}\n"
        "Original report:\n"
        f"{report}\n\n"
        "Produce a final, detailed, plain-language version that explains everything clearly and empathetically."
    )
    messages = [
        {"role": "system", "content": "You are a skilled medical report editor committed to clarity, empathy, and thorough detail."},
        {"role": "user", "content": prompt},
    ]
    return await gpt_call(messages)

# -----------------------------
# Translation Helper Function
# -----------------------------
async def translate_text(text: str, target_language: str) -> str:
    """
    Translate the given text into the target language using the OpenAI API.
    """
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    messages = [
        {"role": "system", "content": "You are a professional translator who accurately translates medical texts."},
        {"role": "user", "content": prompt},
    ]
    return await gpt_call(messages)

# -----------------------------
# ElevenLabs Text-to-Speech Helper Function
# -----------------------------
from pydub import AudioSegment

async def text_to_speech_chunk(text: str, client: httpx.AsyncClient, url: str, headers: dict) -> AudioSegment:
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return AudioSegment.from_file(io.BytesIO(response.content), format="mp3")

async def text_to_speech(text: str) -> bytes:
    """
    Converts long text into speech by splitting it into chunks and then concatenating the audio.
    """
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise Exception("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set in the environment.")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Define a chunk size – adjust as necessary based on your text and API limits
    chunk_size = 500  
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    timeout = httpx.Timeout(120.0, connect=20.0, read=120.0)
    audio_segments = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for chunk in chunks:
            try:
                segment = await text_to_speech_chunk(chunk, client, url, headers)
                audio_segments.append(segment)
            except httpx.ReadTimeout as exc:
                raise HTTPException(
                    status_code=504, 
                    detail="TTS service timed out while processing a segment. Please try again."
                ) from exc
    
    # Concatenate all segments
    combined_audio = audio_segments[0]
    for seg in audio_segments[1:]:
        combined_audio += seg
    
    # Export the combined audio to bytes
    output_buffer = io.BytesIO()
    combined_audio.export(output_buffer, format="mp3")
    return output_buffer.getvalue()

# -----------------------------
# ChatGPT-Based Chat Functionality
# -----------------------------
async def generate_chat_response(report: str, chat_history: List[dict], user_message: str) -> str:
    messages = [
        {"role": "system", "content": f"This conversation is about the following medical report: {report}. (using # for titles and ## for section headings) and bullet lists for key points."}
    ]
    
    # 🟢 Map old roles to OpenAI supported roles ("user" or "assistant")
    for item in chat_history:
        if item["role"] in ["doctor", "patient"]:
            messages.append({
                "role": "user",
                "content": item["content"]
            })
        else:
            messages.append(item)  # assistant stays the same
    
    # Add latest user message
    messages.append({"role": "user", "content": user_message})
    
    return await gpt_call(messages)


# -----------------------------
# Pretty PDF Generation with QR Code using ReportLab and Enhanced Styling
# -----------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import qrcode

def create_pretty_pdf(report_text: str, report_id: str, patient_name: Optional[str] = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Define custom styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Title'],
        fontSize=28,
        textColor=colors.darkblue,
        alignment=1,  # center alignment
        spaceAfter=24
    )
    # Section headings with a lavender background for extra colour
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=colors.darkblue,
        backColor=colors.lavender,
        spaceBefore=12,
        spaceAfter=6,
        leading=24
    )
    # Body text style with increased font size for readability
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.black,
        leading=18,
        spaceAfter=6
    )
    # Call-to-action style for the QR code section
    cta_style = ParagraphStyle(
        'CTAStyle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.darkred,
        alignment=1,
        spaceBefore=12,
        spaceAfter=12
    )
    
    # Clean report_text by removing unwanted markdown symbols
    clean_text = report_text.replace("*", "").replace("_", "")
    
    flowables = []
    
    # Title/Header (no greeting, as duplicate greetings are removed)
    title_text = "Personalized Medical Report"
    if patient_name:
        title_text += f" for {patient_name}"
    header = Paragraph(title_text, header_style)
    flowables.append(header)
    
    # Add the current date only once from the PDF generator
    current_date = datetime.now().strftime("%B %d, %Y")
    date_paragraph = Paragraph(f"Date: {current_date}", body_style)
    flowables.append(date_paragraph)
    flowables.append(Spacer(1, 12))
    flowables.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    flowables.append(Spacer(1, 12))
    
    # Process each line and apply styles dynamically.
    for line in clean_text.split("\n"):
        line = line.strip()
        # Skip any duplicate date line from GPT output
        if line.lower().startswith("date:"):
            continue
        if not line:
            continue
        # If the line ends with a colon or is fully uppercase, treat it as a section heading.
        if line.endswith(":") or line.isupper():
            flowables.append(Paragraph(line, section_heading_style))
        else:
            flowables.append(Paragraph(line, body_style))
        flowables.append(Spacer(1, 6))
    
    flowables.append(Spacer(1, 24))
    # Generate QR Code and add a call-to-action text below it
    chat_url = f"http://localhost:8000/chat/{report_id}"
    qr = qrcode.make(chat_url)
    qr_buffer = io.BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    qr_img = Image(qr_buffer, width=1.5*inch, height=1.5*inch)
    flowables.append(qr_img)
    
    cta_text = "Find out more or ask any questions you have."
    flowables.append(Paragraph(cta_text, cta_style))
    
    doc.build(flowables)
    pdf_output = buffer.getvalue()
    buffer.close()
    return pdf_output

# -----------------------------
# Patient Management Endpoints
# -----------------------------
@app.post("/patients", response_model=Patient)
async def create_patient(name: str = Form(...), nhs_number: str = Form(...)):
    patient_id = str(uuid.uuid4())
    patient = {"id": patient_id, "name": name, "nhs_number": nhs_number}
    patients_db[patient_id] = patient
    save_db(PATIENTS_DB_PATH, patients_db)
    return patient

@app.get("/patients", response_model=List[Patient])
async def list_patients():
    return list(patients_db.values())

@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    patient = patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# -----------------------------
# File Upload & Text Extraction
# -----------------------------
import fitz  # PyMuPDF

@app.post("/patients/{patient_id}/files", response_model=FileRecord)
async def upload_file(patient_id: str, file: UploadFile = File(...)):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    tmp_pdf_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_pdf_path, "wb") as f:
        f.write(await file.read())
    
    pdf_doc = fitz.open(tmp_pdf_path)
    extracted_text = ""
    for page in pdf_doc:
        extracted_text += page.get_text()
    pdf_doc.close()
    os.remove(tmp_pdf_path)
    
    file_id = str(uuid.uuid4())
    file_record = {
        "id": file_id,
        "patient_id": patient_id,
        "filename": file.filename,
        "extracted_text": extracted_text,
    }
    files_db[file_id] = file_record
    save_db(FILES_DB_PATH, files_db)
    return file_record

# -----------------------------
# Multi-Agent Report Generation Endpoint
# -----------------------------
@app.post("/patients/{patient_id}/files/{file_id}/generate-report", response_model=Report)
async def generate_report(
    patient_id: str, 
    file_id: str, 
    complexity: str = Form(...),
    target_language: str = Form("en")  # default to English if not provided
):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    file_record = files_db.get(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Gather context from all files for this patient
    all_texts = [f["extracted_text"] for f in files_db.values() if f["patient_id"] == patient_id]
    combined_context = "\n\n".join(all_texts)
    combined_context += f"\n\nLatest file details:\n{file_record['extracted_text']}"
    
    report_id = str(uuid.uuid4())
    report = {
        "id": report_id,
        "patient_id": patient_id,
        "file_id": file_id,
        "complexity": complexity,
        "initial_report": None,
        "critique_feedback": None,
        "final_report": None,
    }
    reports_db[report_id] = report

    # Generate the report in English first.
    initial = await agent_1_generate_report(combined_context, complexity)
    critique = await agent_2_critique(initial)
    refined = await agent_3_refine(initial, critique)
    
    # If the target language is not English, translate the refined report.
    if target_language.lower() not in ["en", "english"]:
        lang_map = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}
        target_language_full = lang_map.get(target_language.lower(), target_language)
        translated = await translate_text(refined, target_language_full)
        refined = translated

    # Save all outputs
    report["initial_report"] = initial
    report["critique_feedback"] = critique
    report["final_report"] = refined

    save_db(REPORTS_DB_PATH, reports_db)
    return report


# -----------------------------
# PDF Download Endpoint (Pretty PDF with QR Code)
# -----------------------------
@app.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    report = reports_db.get(report_id)
    if not report or not report.get("final_report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    patient_id = report.get("patient_id")
    patient = patients_db.get(patient_id)
    patient_name = patient.get("name") if patient else None
    
    pretty_pdf = create_pretty_pdf(report["final_report"], report_id, patient_name)
    return StreamingResponse(
        io.BytesIO(pretty_pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.pdf"}
    )

# -----------------------------
# Translated Report Download Endpoint
# -----------------------------
@app.get("/reports/{report_id}/translate")
async def download_translated_report(report_id: str, target_language: str = Query(...)):
    report = reports_db.get(report_id)
    if not report or not report.get("final_report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Convert language code to full language name
    lang_map = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}
    target_language_full = lang_map.get(target_language, target_language)
    
    translated_text = await translate_text(report["final_report"], target_language_full)
    
    patient_id = report.get("patient_id")
    patient = patients_db.get(patient_id)
    patient_name = patient.get("name") if patient else None

    translated_pdf = create_pretty_pdf(translated_text, report_id, patient_name)
    return StreamingResponse(
        io.BytesIO(translated_pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=translated_report_{report_id}.pdf"}
    )


# -----------------------------
# Text-to-Speech (TTS) Endpoint for Report Audio
# -----------------------------
@app.get("/reports/{report_id}/tts")
async def download_report_tts(report_id: str):
    report = reports_db.get(report_id)
    if not report or not report.get("final_report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    audio_bytes = await text_to_speech(report["final_report"])
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.mp3"}
    )

@app.get("/patients/{patient_id}/files")
async def list_patient_files(patient_id: str):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient_files = [f for f in files_db.values() if f["patient_id"] == patient_id]
    return patient_files

# -----------------------------
# Chat Endpoint for Report-Related Questions
# -----------------------------
@app.post("/chat/{report_id}")
async def chat(report_id: str, message: ChatMessage):
    report = reports_db.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    chat_history = chat_db.get(report_id, [])

    # 🔵 Map "doctor"/"patient" to "user"
    openai_role = "user"  # You could also differentiate them by using custom metadata, not the role.

    chat_history.append({"role": openai_role, "content": message.message})

    response_text = await generate_chat_response(
        report["final_report"], 
        chat_history, 
        message.message
    )

    chat_history.append({"role": "assistant", "content": response_text})
    chat_db[report_id] = chat_history
    save_db(CHAT_DB_PATH, chat_db)
    
    return {"response": response_text}

@app.post("/emailReport/{report_id}")
async def send_email_with_pdf(
    report_id: str,
    background_tasks: BackgroundTasks,
):
    # Get the report
    report = reports_db.get(report_id)
    if not report or not report.get("final_report"):
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Get patient name if available
    patient_id = report.get("patient_id")
    patient = patients_db.get(patient_id)
    patient_name = patient.get("name") if patient else None
    
    # Generate PDF bytes
    pdf_bytes = create_pretty_pdf(report["final_report"], report_id, patient_name)

    # ✅ Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)  # Write the generated PDF bytes to the file
        temp_pdf_path = temp_pdf.name  # Get the temp file path

    # Create email message with the file path
    message = MessageSchema(
        subject="SimplyMed Report",
        recipients=["wengloo135@gmail.com"],
        body="This is your health report. May you live to 300 years old XD",
        subtype="plain",
        attachments=[temp_pdf_path]  
    )

    # Send email in the background
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)

    return {"message": "Email with PDF attachment is being sent"}


# -----------------------------
# Main entry point to run with uvicorn
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
