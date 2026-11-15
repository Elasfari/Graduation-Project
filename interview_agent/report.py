from interview_agent.technical_depth_analysis import run_exam_pipeline
from interview_agent.speech_to_text import transcribe_mp3
from interview_agent.emotion import analyze_emotional_state
from interview_agent.pixtral import analyze_with_pixtral_model
from interview_agent.sentiment_analysis import format_analysis_report
from interview_agent.sentiment_analysis import analyze_interview_advanced
from datetime import datetime
import os
import requests
from openai import OpenAI




def beautify_analysis_with_llm(raw_sentiment: str, raw_pixtral: str) -> str:
    """
    Uses OpenAI GPT-4 to clean and structure sentiment and pixtral analysis results.
    Returns a polished, professional unified report string.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
You are a professional technical writer. Your task is to take a raw interview analysis report (formatted in markdown-style text, with \\n line breaks, metrics, and noise) and turn it into a beautiful, structured, readable executive report.

### Guidelines:
- Keep all major sections and subheadings, but format them clearly and professionally.
- Use appropriate emojis for headings (📊, 😌, 🧠, ✅, ⚠, 🔁, ❓, etc.) for visual clarity.
- Convert bullet points into paragraphs or bullet sections cleanly.
- Rephrase any percentages into context-aware insights.
- Remove markdown noise: *, **, hashtags, stray punctuation, etc.
- Ensure smooth transitions and formal tone suitable for a hiring manager or executive.

### Raw INTERVIEW REPORT:
{raw_sentiment}

### Raw PIXTRAL INSIGHTS:
{raw_pixtral}
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a professional report formatter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()



def analyze_video_workflow(video_path: str) -> dict:
    """
    Wrapper function to perform a complete analysis on a video.
    
    Workflow:
    1. Transcribe video audio using transcribe_mp3.
    2. Analyze sentiment from the transcribed text.
    3. Analyze emotional state from the video.
    
    Returns:
        dict: A dictionary containing the transcript, sentiment analysis, and emotional insights.
    """
    # Step 1: Transcription
    transcript = transcribe_mp3(video_path)

    sentiment_result = analyze_interview_advanced(transcript)
    emotional_state_result = analyze_emotional_state(video_path, 2)
    raw_sentiment = format_analysis_report(sentiment_result)
    raw_pixtral = analyze_with_pixtral_model("interview_agent/output")

    beautified_report = beautify_analysis_with_llm(raw_sentiment, raw_pixtral)

    return beautified_report
    

    # update_supabase_profile(result)

    

# if __name__ == "__main__":
#     video_path = "interview_agent/Emaraty.mp4"
#     result = analyze_video_workflow(video_path)

#     print("Analysis complete. Generating PDF report...")
#     pdf_path = generate_pdf_report(result)
#     print(f"Professional PDF report saved to: {pdf_path}")