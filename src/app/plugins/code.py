import os
import subprocess

from semantic_kernel.functions.kernel_function_decorator import kernel_function


class CodeRunnerPlugin:
    """
    CodeRunnerPlugin
    """

    @kernel_function(
        name="ExecuteCode",
        description="Execute python code based on preset docker image. Takes the code to run and the path to the uploaded file as input."
    )
    def run_generated_code(self, generated_code, uploaded_file_path):
        # Write the generated code to a temporary file
        code_file = 'analysis.py'
        with open(code_file, 'w') as f:
            f.write(generated_code)

        # Build the docker run command with appropriate options
        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=256m",
            "--cpus=0.5",
            "-v", f"{uploaded_file_path}:/app/uploaded_file.csv:ro",
            "-v", f"{os.path.abspath(code_file)}:/app/generated_analysis.py:ro",
            "code-runner",
            "python", "/app/analysis.py"
        ]

        # Execute the container and capture output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", "Execution timed out."
        finally:
            # Optionally, clean up the temporary file
            os.remove(code_file)
