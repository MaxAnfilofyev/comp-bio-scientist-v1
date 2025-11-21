import subprocess
from pathlib import Path
from typing import Dict, Any

from ai_scientist.tools.base_tool import BaseTool


class ManuscriptReaderTool(BaseTool):
    """
    Load manuscript text from a PDF (via pdftotext) or a plain text/Markdown file.
    Returns a dict with 'path' and 'text'. If pdf, tries to use pdftotext; otherwise reads as text.
    """

    def __init__(
        self,
        name: str = "ReadManuscript",
        description: str = (
            "Read a manuscript draft from PDF (pdftotext) or txt/md and return its text. "
            "Use to review current or previous drafts."
        ),
    ):
        parameters = [
            {"name": "path", "type": "str", "description": "Path to PDF or text/markdown file."},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manuscript not found: {path}")

        text = ""
        if p.suffix.lower() == ".pdf":
            # Try pdftotext
            try:
                out_path = p.with_suffix(".txt")
                subprocess.run(
                    ["pdftotext", "-layout", str(p), str(out_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if out_path.exists():
                    text = out_path.read_text()
            except Exception as e:
                text = ""
                raise e
        else:
            text = p.read_text()

        return {"path": str(p), "text": text}
