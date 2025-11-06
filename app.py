from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

app = Flask(__name__)
CORS(app)

# Load .env and configure OpenAI API (read from environment)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. API calls will fail unless you set it.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_response():
    data = request.json or {}
    stage = data.get('stage', 'start')
    user_response = data.get('response', '')
    role = data.get('role', 'candidate')
    history = data.get('history', [])
    userName = data.get('userName', '')

    # If no model client configured, inform the frontend that model is not working
    if client is None:
        return jsonify({'success': False, 'error': 'Model not working'}), 200

    # System prompt defines interview behavior and output contract; encourage varied questions and brief feedback
    system_prompt = (
        "You are an extremely professional, human-like interviewer for technical roles. Behavior requirements:\n"
        "- Vary your questions: start with simple conceptual prompts (definitions), then ask about differences or comparisons, then application-level problems, and finally harder design/optimization or edge-case questions as the candidate demonstrates understanding.\n"
        "- Do NOT get stuck on a single topic: if the candidate answers, acknowledge briefly (1 short sentence), optionally give a quick correction or hint if the answer is off, then proceed to the next logical question.\n"
        "- If the candidate gives an incomplete or unclear answer, ask one short follow-up clarification before moving on.\n"
        "- Keep tone professional, polite, and focused; behave like a live interviewer who tailors questions from the candidate's prior answers.\n"
        "Staged flow (use these stage names): GREET -> name -> intro -> education -> questions -> end.\n"
        "Return format: Return ONLY a valid JSON object and nothing else. Required keys: {\n"
        "  \"question\": <string, the next interviewer prompt to display to the candidate>,\n"
        "  \"next_stage\": <string, next stage name>,\n"
        "  \"success\": true\n"
        "}\n"
        "Optional fields you may include: \"feedback\" (a short sentence giving a brief comment on the candidate's last answer), \"followup\" (a short clarification question to ask before the main next question).\n"
        "Important: When in QUESTIONS stage, alternate question difficulty from simple to medium to hard; use the candidate's earlier answers and role to craft realistic, specific prompts. If the user says 'end' or 'stop', set next_stage='end' and return a closing 'question' string that is a friendly goodbye message."
    )

    # Build a short transcript (last 10 messages) to give context
    hist_items = history[-10:] if history else []
    history_text = "\n".join([f"{item.get('type','')}: {item.get('text','')}" for item in hist_items])

    user_msg = (
        f"Role: {role}\nStage: {stage}\nLast user response: {user_response}\nHistory:\n{history_text}\n\n"
        "Based on the stage and history, generate the next interviewer question and indicate the next_stage in JSON as described."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.2
        )

        text = resp.choices[0].message.content.strip()
        # Extract JSON object from model response
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return jsonify({'success': False, 'error': 'Model did not return JSON'}), 200

        result = json.loads(m.group())
        # Validate
        if not isinstance(result, dict) or 'question' not in result or 'next_stage' not in result:
            return jsonify({'success': False, 'error': 'Model returned invalid JSON'}), 200

        # Keep backward compatibility with frontend keys
        out = {
            'success': True,
            'nextQuestion': result.get('question'),
            'stage': result.get('next_stage')
        }
        # include any extra fields the model returned
        for k, v in result.items():
            if k not in ('question', 'next_stage'):
                out[k] = v
        return jsonify(out)

    except Exception as e:
        print(f"Model error in process_response: {e}")
        return jsonify({'success': False, 'error': 'Model not working'}), 200


@app.route('/api/evaluate', methods=['POST'])
def evaluate_interview():
    data = request.json or {}
    history = data.get('history', [])
    role = data.get('role', 'candidate')
    userName = data.get('userName', 'candidate')

    if client is None:
        return jsonify({'success': False, 'error': 'Model not working'}), 200

    # Build transcript for the model
    qa_pairs = []
    # skip markers for answers we don't want to include in history
    skip_markers = ['(Question skipped', '(No response', '(Question skipped - no start)', '(Question skipped - silence)', '(Question skipped - not prepared)', '(No response - Timeout)']
    for i in range(len(history)):
        if history[i].get('type') == 'bot':
            question = history[i].get('text', '')
            # skip administrative bot messages like the too-many-skips closing message
            if 'Too many skipped' in question:
                continue
            answer = ''
            if i + 1 < len(history) and history[i+1].get('type') == 'user':
                answer = history[i+1].get('text', '')
            # if answer is empty or a skip marker, do not include this QA pair in the transcript
            if not answer or any(marker in answer for marker in skip_markers):
                continue
            qa_pairs.append({'question': question, 'answer': answer})

    transcript = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])

    # Build the evaluation prompt without f-string braces collisions
    prompt_prefix = (
        "You are a strict, fair interviewer evaluator. Evaluate the candidate's performance for role "
        + str(role) + " using ONLY the transcript below.\n\n"
        "TRANSCRIPT:\n"
    )

    prompt_example = (
        "Return ONLY a single JSON object (no surrounding text). The JSON MUST have these keys and types:\n"
        "- score: integer 0-100 (the total score). This MUST equal the sum of the breakdown fields.\n"
        "- breakdown: object with integer fields: technical, problem_solving, communication, experience, critical_thinking. These are numeric points that sum exactly to score.\n"
        "- maxBreakdown: object with the integer maxima for each field (use these maxima): technical:35, problem_solving:25, communication:20, experience:15, critical_thinking:5\n"
        "- strengths: list of short strings (3-6 items maximum describing what the candidate did well).\n"
        "- weaknesses: list of short strings (3-6 items maximum describing what the candidate could improve).\n"
        "- suggestions: list of short actionable suggestions (3-6 items).\n"
        "- overall: short summary string (1-2 sentences).\n\n"
        "Be concise and honest. Ensure numeric fields are integers and sums are consistent. Example shape:\n"
        "{\n  \"score\": 72,\n  \"breakdown\": {\"technical\": 25, \"problem_solving\": 20, \"communication\": 15, \"experience\": 8, \"critical_thinking\": 4},\n  \"maxBreakdown\": {\"technical\":35, \"problem_solving\":25, \"communication\":20, \"experience\":15, \"critical_thinking\":5},\n  \"strengths\": [\"...\"],\n  \"weaknesses\": [\"...\"],\n  \"suggestions\": [\"...\"],\n  \"overall\": \"...\"\n}\n"
    )

    prompt = prompt_prefix + transcript + "\n\n" + prompt_example

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a strict, fair interviewer evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            # Model did not return JSON — return a safe fallback evaluation so UI shows history
            fallback = {
                'success': True,
                'score': 0,
                'breakdown': {'technical':0,'problem_solving':0,'communication':0,'experience':0,'critical_thinking':0},
                'maxBreakdown': {'technical':35,'problem_solving':25,'communication':20,'experience':15,'critical_thinking':5},
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'overall': 'Model did not return a valid evaluation. Showing interview transcript only.',
                'qaList': qa_pairs,
                'model_error': 'did not return JSON'
            }
            return jsonify(fallback), 200

        evaluation = json.loads(m.group())

        # Basic validation of evaluation structure
        required_keys = ('score', 'breakdown', 'maxBreakdown', 'strengths', 'weaknesses', 'suggestions', 'overall')
        if not all(k in evaluation for k in required_keys):
            print("Evaluation missing required keys:", evaluation.keys())
            fallback = {
                'success': True,
                'score': 0,
                'breakdown': {'technical':0,'problem_solving':0,'communication':0,'experience':0,'critical_thinking':0},
                'maxBreakdown': {'technical':35,'problem_solving':25,'communication':20,'experience':15,'critical_thinking':5},
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'overall': 'Model returned incomplete evaluation. Showing interview transcript only.',
                'qaList': qa_pairs,
                'model_error': 'incomplete_keys'
            }
            return jsonify(fallback), 200

        # Validate numeric types and consistency
        try:
            score = int(evaluation['score'])
            breakdown = evaluation['breakdown']
            maxbd = evaluation['maxBreakdown']
            bd_keys = ['technical', 'problem_solving', 'communication', 'experience', 'critical_thinking']
            if not all(k in breakdown for k in bd_keys):
                raise ValueError('breakdown missing fields')
            bd_values = [int(breakdown[k]) for k in bd_keys]
            if sum(bd_values) != score:
                raise ValueError('breakdown does not sum to score')
            expected_max = {'technical':35, 'problem_solving':25, 'communication':20, 'experience':15, 'critical_thinking':5}
            # If model didn't provide exact maxima, set the expected maxima
            if not isinstance(maxbd, dict) or not all(int(maxbd.get(k, -1)) == expected_max[k] for k in expected_max):
                evaluation['maxBreakdown'] = expected_max

        except Exception as e:
            print("Evaluation validation error:", e)
            fallback = {
                'success': True,
                'score': 0,
                'breakdown': {'technical':0,'problem_solving':0,'communication':0,'experience':0,'critical_thinking':0},
                'maxBreakdown': {'technical':35,'problem_solving':25,'communication':20,'experience':15,'critical_thinking':5},
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'overall': 'Model returned invalid evaluation types. Showing interview transcript only.',
                'qaList': qa_pairs,
                'model_error': 'invalid_types'
            }
            return jsonify(fallback), 200

        evaluation['success'] = True
        # Attach qaList for frontend display
        evaluation['qaList'] = qa_pairs
        return jsonify(evaluation)
    except Exception as e:
        print(f"Model error in evaluate_interview: {e}")
        return jsonify({'success': False, 'error': 'Model not working'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)