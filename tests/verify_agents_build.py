import sys
import os
sys.path.append(os.getcwd())
from ai_scientist.orchestrator.agents import build_team

dirs = {"base": "/tmp", "results": "/tmp/results"}
idea = {"Title": "Test", "Abstract": "Abs", "Experiments": [], "Risk Factors and Limitations": []}
try:
    team = build_team("model", idea, dirs)
    print("Successfully built team!")
except Exception as e:
    print(f"Failed to build team: {e}")
    sys.path.append("/Users/maxa/AI-Scientist-v2")
    import traceback
    traceback.print_exc()
