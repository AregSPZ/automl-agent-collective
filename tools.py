from langchain_core.tools import tool

# Code execution tool
@tool(description="""Use this tool when you're asked to write and execute Python code.""")
def execute_code(code: str) -> str:
    print("Tool triggered: execute_code")
    # use try-except block to catch errors in code
    try:
        import io, contextlib
        # make a string buffer
        buffer = io.StringIO()
        # execute the code with redirecting the stdout to the buffer
        with contextlib.redirect_stdout(buffer):
            exec(code, globals())
        # return stdout if everything worked out
        return buffer.getvalue()
    except Exception as e:
        return f"Error: {e}"
    

# mapping of tools agents use
toolkit = {
    "problem_framer": [execute_code],
    "data_preprocessor": [execute_code],
    "model_selector": [],
    "evaluator": [execute_code],
}