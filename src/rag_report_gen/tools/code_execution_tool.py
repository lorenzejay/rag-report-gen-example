import os
from typing import Type
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
from e2b_code_interpreter import Sandbox
import base64


class E2BCodeInterpreterSchema(BaseModel):
    """Input schema for the CodeInterpreterTool, used by the agent."""

    code: str = Field(
        ...,
        description="Python3 code used to run in the Jupyter notebook cell. Non-standard packages are installed by appending !pip install [packagenames] and the Python code in one single code block.",
    )


class CodeInterpreterTool(BaseTool):
    """

    This is a tool that runs arbitrary code in a Python Jupyter notebook.

    It uses E2B to run the notebook in a secure cloud sandbox.

    It requires an E2B_API_KEY to create a sandbox.

    """

    name: str = "code_interpreter"

    description: str = "Execute Python code in a Jupyter notebook cell and return any rich data (eg charts), stdout, stderr, and errors."

    args_schema: Type[BaseModel] = E2BCodeInterpreterSchema

    _code_interpreter_tool: Sandbox | None = None

    def _run(self, code: str) -> str:
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        sandbox = Sandbox(api_key=os.environ["E2B_API_KEY"])

        execution = sandbox.run_code(code)
        first_result = execution.results[0]
        if first_result.png:
            with open("outputs/financial_report_graph.png", "wb") as f:
                f.write(base64.b64decode(first_result.png))

        return "Chart saved as financial_report_graph.png"
