from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
from datetime import datetime
from typing import TypedDict, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain and LangGraph imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    
    GROQ_KEY = os.environ.get('GROQ_API_KEY')
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_KEY,
        temperature=0.7,
        max_tokens=150
    ) if GROQ_KEY else None
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    logger.error(f"Import error: {e}")
    llm = None
    LANGGRAPH_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Patient persona for the AI
PATIENT_PERSONA = """You are a patient with a migraine. Answer questions naturally and briefly. 
Only reveal symptoms when asked. Be conversational and realistic."""

# State definition for LangGraph
class PatientState(TypedDict):
    messages: List[Dict]
    symptoms_revealed: int
    condition: str
    treatment_accepted: bool
    conversation_stage: str
    session_id: str

# Node 1: Analyze what stage of conversation we're in
def analyze_stage(state: PatientState) -> PatientState:
    if not state["messages"]:
        state["conversation_stage"] = "initial"
        return state
    
    last_message = state["messages"][-1]["content"].lower()
    
    # Check what the doctor is asking about
    if any(word in last_message for word in ["prescribe", "medication", "treatment", "take"]):
        state["conversation_stage"] = "treatment"
    elif any(word in last_message for word in ["pain", "hurt", "ache"]):
        state["conversation_stage"] = "pain"
    elif any(word in last_message for word in ["symptom", "feel", "experiencing"]):
        state["conversation_stage"] = "symptoms"
    elif any(word in last_message for word in ["when", "start", "began"]):
        state["conversation_stage"] = "timeline"
    else:
        state["conversation_stage"] = "general"
    
    return state

# Node 2: Generate patient response
def generate_response(state: PatientState) -> PatientState:
    # Fallback responses if API is unavailable
    fallbacks = {
        "treatment": "Thank you doctor. How often should I take it?",
        "pain": "It's pretty bad, around 7 out of 10. Left side of my head.",
        "symptoms": "I'm sensitive to light and feeling nauseous. Sometimes I see weird patterns.",
        "timeline": "Started about three days ago. Getting worse.",
        "general": "I'm worried about this headache. It's really affecting me."
    }
    
    try:
        if llm and GROQ_KEY:
            # Get recent conversation
            recent = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
            
            # Build messages for LLM
            msgs = [SystemMessage(content=PATIENT_PERSONA)]
            for msg in recent:
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg['content']))
            
            # Get AI response
            response = llm.invoke(msgs)
            reply = response.content.strip()
        else:
            reply = fallbacks.get(state['conversation_stage'], "Could you repeat that?")
    
    except Exception as e:
        logger.error(f"LLM error: {e}")
        reply = fallbacks.get(state['conversation_stage'], "I didn't catch that.")
    
    # Add response to messages
    state["messages"].append({
        "role": "assistant",
        "content": reply,
        "timestamp": datetime.now().isoformat()
    })
    
    return state

# Node 3: Update consultation state
def update_state(state: PatientState) -> PatientState:
    stage = state["conversation_stage"]
    
    if stage == "treatment":
        state["treatment_accepted"] = True
        state["condition"] = "Treatment Prescribed"
    elif stage in ["symptoms", "pain", "timeline"]:
        state["symptoms_revealed"] = min(state.get("symptoms_revealed", 0) + 1, 10)
        state["condition"] = "In Consultation"
    
    return state

# Create LangGraph workflow
def create_workflow():
    workflow = StateGraph(PatientState)
    
    workflow.add_node("analyze_stage", analyze_stage)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("update_state", update_state)
    
    workflow.add_edge("analyze_stage", "generate_response")
    workflow.add_edge("generate_response", "update_state")
    workflow.add_edge("update_state", END)
    
    workflow.set_entry_point("analyze_stage")
    
    return workflow.compile()

# Initialize workflow
try:
    if LANGGRAPH_AVAILABLE:
        patient_graph = create_workflow()
    else:
        patient_graph = None
except Exception as e:
    logger.error(f"Failed to compile workflow: {e}")
    patient_graph = None

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Patient Chatbot API",
        "groq_configured": bool(GROQ_KEY),
        "langgraph_available": LANGGRAPH_AVAILABLE
    })

@app.route('/api/init', methods=['POST'])
def init_session():
    try:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        state = {
            "messages": [{
                "role": "assistant",
                "content": "Hi doctor, I'm not feeling well. I've had this headache for a few days now.",
                "timestamp": datetime.now().isoformat()
            }],
            "symptoms_revealed": 1,
            "condition": "Initial Consultation",
            "treatment_accepted": False,
            "conversation_stage": "greeting",
            "session_id": session_id
        }
        
        return jsonify({
            "session_id": session_id,
            "message": state["messages"][0],
            "patient_info": {
                "condition": state["condition"],
                "symptoms_revealed": state["symptoms_revealed"],
                "treatment_accepted": False
            },
            "state": state
        })
    
    except Exception as e:
        logger.error(f"Init error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        message = data.get('message', '').strip()
        state = data.get('state', {})
        session_id = data.get('session_id', 'unknown')
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        if not state or 'messages' not in state:
            return jsonify({"error": "Invalid state"}), 400
        
        # Add doctor's message
        state["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        state["session_id"] = session_id
        
        # Run workflow
        if patient_graph:
            result = patient_graph.invoke(state)
        else:
            result = state
            result = analyze_stage(result)
            result = generate_response(result)
            result = update_state(result)
        
        return jsonify({
            "message": result["messages"][-1],
            "patient_info": {
                "condition": result.get("condition", "In Consultation"),
                "symptoms_revealed": result.get("symptoms_revealed", 0),
                "treatment_accepted": result.get("treatment_accepted", False)
            },
            "state": result,
            "using_langgraph": patient_graph is not None
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500