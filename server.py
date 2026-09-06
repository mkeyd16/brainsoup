import os
import sys
import torch
from flask import Flask, request, jsonify
from safetensors.torch import load_file

# --------------------------------------------------
# BrainSoup - MicroMixer-4-50K local API
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "MicroMixer-4")
CHECKPOINT = os.path.join(BASE_DIR, "fmsp_epoch_9.safetensors")

# Allow imports from the MicroMixer repository
sys.path.insert(0, REPO_DIR)

from src.model_v87_final import MicroMixerV87Final, v87_final_50k
from src.fmsp import attach_adapter
from src.tokenizer import ByteTokenizer


print("=" * 60)
print("                    BRAIN SOUP")
print("=" * 60)
print("Loading MicroMixer-4 50K...")

# --------------------------------------------------
# Load model
# --------------------------------------------------

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

cfg = v87_final_50k()

model = MicroMixerV87Final(cfg)

attach_adapter(
    model,
    d_model=cfg.d_model,
    rank=16
)

state_dict = load_file(CHECKPOINT)

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

tokenizer = ByteTokenizer()

print("Model loaded successfully.")
print(f"Checkpoint: {CHECKPOINT}")
print("API: http://127.0.0.1:5000")
print("=" * 60)


# --------------------------------------------------
# Flask
# --------------------------------------------------

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": "MicroMixer-4-50K"
    })


@app.post("/chat")
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be JSON."
        }), 400

    message = data.get("message")

    if not isinstance(message, str):
        return jsonify({
            "error": "Missing 'message' string."
        }), 400

    message = message.strip()

    if not message:
        return jsonify({
            "error": "Message cannot be empty."
        }), 400

    # Prevent accidentally huge prompts
    message = message[:500]

    # MicroMixer was trained using this conversational format.
    prompt = f"User: {message}\n\nAssistant: "

    ids = tokenizer.encode(prompt)

    # The official model example removes a trailing EOS token.
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]

    input_ids = torch.tensor([ids])

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0].tolist()
    )

    # Only return the generated Assistant portion.
    if "Assistant:" in response:
        response = response.split("Assistant:", 1)[1]

    # Prevent accidental continuation into another User message.
    if "User:" in response:
        response = response.split("User:", 1)[0]

    response = " ".join(response.split())

    return jsonify({
        "response": response
    })


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        threaded=True,
        use_reloader=False
    )