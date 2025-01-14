import os
import warnings
from flask import Flask, request, jsonify
import whisper
import openai
from flask_cors import CORS

warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

app = Flask(__name__)
CORS(app)

def audio_to_text(audio_file_path):
    try:
        if not os.path.exists(audio_file_path):
            return "Error: File not found."

        model = whisper.load_model("base")  
        result = model.transcribe(audio_file_path, language="en", task="transcribe", fp16=False)
        transcription = result["text"]
        return transcription.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_reply(user_query):
    openai.api_key = os.getenv('API_KEY')# Replace with your OpenAI API key

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides human-like responses."},
                {"role": "user", "content": f"This is the user query: {user_query}. Generate a suggestible response analysing the users intent in the query and only give me back the response and nothing else. Try to act like human."},
            ],
            temperature=0.7,
            max_tokens=100,
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/process-audio/", methods=['POST', 'GET'])
def process_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        audio_file_path = f"temp_{file.filename}"

        file.save(audio_file_path)

        transcription = audio_to_text(audio_file_path)

        if "Error" in transcription:
            return jsonify({"error": transcription}), 400

        response = generate_reply(transcription)

        os.remove(audio_file_path)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
