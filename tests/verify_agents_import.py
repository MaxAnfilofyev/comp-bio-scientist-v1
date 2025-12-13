
import sys
import os
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import build_team from ai_scientist.orchestrator.agents...")
    from ai_scientist.orchestrator.agents import build_team
    print("✅ Successfully imported build_team.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
