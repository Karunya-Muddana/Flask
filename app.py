# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from tools import google_search, wiki_search, fetch_page_content
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# === Gemini Setup ===
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini key loaded ✅")

# === Tool Definitions ===
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="google_search",
            description="Search Google using Custom Search API",
            parameters=types.Schema(type="object", properties={"query": {"type": "string"}}, required=["query"])
        ),
        types.FunctionDeclaration(
            name="wiki_search",
            description="Search Wikipedia for relevant articles",
            parameters=types.Schema(type="object", properties={"query": {"type": "string"}}, required=["query"])
        ),
        types.FunctionDeclaration(
            name="fetch_page_content",
            description="Extract readable content from a webpage",
            parameters=types.Schema(type="object", properties={"url": {"type": "string"}}, required=["url"])
        ),
    ])
]
config = types.GenerateContentConfig(tools=tools)

# === Persona Conditioning ===
KARUNYA_PERSONA = """
You are an AI assistant representing **Karunya Muddana**, a male Backend and AI Developer from Hyderabad, India.
He is a Computer Science student passionate about Artificial Intelligence, MLOps, and Full-Stack Development.

Karunya specializes in building **AI-powered systems**, including:
- Agentic AI pipelines and multi-agent systems.
- RAG (Retrieval-Augmented Generation) applications.
- Flask-based backend architectures integrated with LLM APIs.
- Interactive portfolios and intelligent user interfaces using React, Vite, and TypeScript.

He has hands-on experience with **Python, Flask, React, TypeScript, and automation systems**, and enjoys combining AI with strong design and system-level problem-solving.
He actively documents his projects on LinkedIn through detailed technical breakdowns and development logs that focus on AI engineering, prompt architecture, and autonomous agent design.

His GitHub (**Karunya-Muddana**) contains 19 repositories featuring:
- Flask and Python AI tools (e.g., Gemini Multi-Tool Agent integrating Google Search, Gmail, and Web Scraping).
- Machine Learning projects like Linear Regression, Stock Prediction, and Fine-Tuning with Llama 3.
- Full-stack applications (CMDportfolio, Futuristic Personal Portfolio).
- Streamlit analytics apps (CBSE Result Analysis).
- Automation and utility scripts (QR Scanner, Drugstore Manager, License Plate Detector).

When representing him:
- Always use masculine pronouns (he/him).
- Maintain a professional, factual, and composed tone — similar to how an AI engineer would explain his own work.
- Highlight logic, design thinking, and technical depth rather than grades or academic numbers.
"""


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Combine persona + user message
        combined_prompt = f"{KARUNYA_PERSONA}\n\nUser query: {message}"

        history = [
            types.Content(role="user", parts=[types.Part(text=combined_prompt)])
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=config
        )

        # === Tool Call Handling ===
        while response.function_calls:
            tool_outputs = []
            for call in response.function_calls:
                func_name = call.name
                args = dict(call.args)
                try:
                    if func_name == "google_search":
                        result = google_search(**args)
                    elif func_name == "wiki_search":
                        result = wiki_search(**args)
                    elif func_name == "fetch_page_content":
                        result = fetch_page_content(**args)
                    else:
                        result = f"Unknown tool: {func_name}"
                except Exception as e:
                    result = f"Error executing {func_name}: {e}"

                tool_outputs.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=func_name,
                        response={"content": result}
                    ))
                )

            history.append(types.Content(role="tool", parts=tool_outputs))
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history,
                config=config
            )

        # === Extract Text ===
        final_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                final_text += part.text

        return jsonify({"response": final_text.strip()})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Karunya AI Agent with Gemini Tools is live."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
