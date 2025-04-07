import os
import subprocess
import sys
import tempfile
import asyncio
import json
from dotenv import load_dotenv
import semantic_kernel as sk
import jinja2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import logger
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.exceptions import ServiceResponseException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE)

COSMOS_DB_NAME = os.getenv("COSMOS_QNA_NAME", "mydb")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://myendpoint.documents.azure.com:443/")
COSMOS_ASSEMBLY_TABLE = os.getenv("COSMOS_ASSEMBLY_TABLE", "assembly")
AZURE_MODEL_KEY = os.getenv("AZURE_MODEL_KEY", "")
AZURE_MODEL_URL = os.getenv("AZURE_MODEL_URL", "")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "prompts")
JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR))

AVAILABLE_MODELS: list[AzureChatCompletion] = [
    AzureChatCompletion(
        service_id="default",
        api_key=AZURE_MODEL_KEY,
        deployment_name="gpt-4o",
        endpoint=AZURE_MODEL_URL
    ),
    AzureChatCompletion(
        service_id="mini",
        api_key=AZURE_MODEL_KEY,
        deployment_name="gpt-4o-mini",
        endpoint=AZURE_MODEL_URL
    ),
    AzureChatCompletion(
        service_id="reasoning",
        api_key=AZURE_MODEL_KEY,
        deployment_name="o3-mini",
        endpoint=AZURE_MODEL_URL,
        api_version="2024-12-01-preview",
    )
]

@kernel_function(
    name="ExecuteCode",
    description="Execute python code using the current interpreter. The code reads a CSV file (path provided as argument), performs analysis, and outputs results. The output is captured and written to a PDF file."
)
def run_generated_code(generated_code):
    """
    Executes the provided Python code using the current interpreter.
    
    The generated code is expected to have the CSV data embedded within it.
    The function captures the code's stdout and stderr and writes the stdout 
    to a PDF file named 'analysis_output.pdf'.
    
    Returns:
        tuple: (stdout, stderr, pdf_file_path)
    """
    # Write the generated code to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        code_file = tmp.name
        tmp.write(generated_code)
    
    # Build the command to run the code using the current Python interpreter.
    cmd = [sys.executable, os.path.abspath(code_file)]
    
    try:
        # Execute the code and capture its output.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Execution timed out."
    finally:
        # Clean up the temporary file.
        os.remove(code_file)
    
    # Create a PDF file with the captured stdout.
    pdf_file = os.path.join(os.getcwd(), "analysis_output.pdf")
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    for line in stdout.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    
    return stdout, stderr, pdf_file

async def run_code_in_docker(prompt: str, csv: str) -> None:
    kernel = sk.Kernel()
    for service in AVAILABLE_MODELS:
        try:
            kernel.add_service(service)
        except Exception as e:
            logger.error(f"Failed to add service {service.service_id}: {e}")
    # Register the modified plugin with the name "ExecuteCode"
    kernel.add_plugin(plugin=run_generated_code, plugin_name="ExecuteCode")
    
    settings = AzureChatPromptExecutionSettings(
        service_id="default",
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke=True)
    
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="reasoning_agent",
        instructions=prompt,
        arguments=KernelArguments(settings=settings)
    )
    
    chat = ChatHistory()
    chat.add_message(ChatMessageContent(role=AuthorRole.SYSTEM, content=prompt))
    chat.add_message(ChatMessageContent(role=AuthorRole.USER, content=f'{csv}'))
    while True:
        async for message in agent.invoke(messages=chat.messages):  # type: ignore
            try:
                # Try parsing the message content as JSON.
                content_dict = json.loads(str(message.content))
            except json.JSONDecodeError:
                # If parsing fails, assume this is the final answer.
                print(message.content)
                break

            if "function_call" in content_dict:
                # Instead of manually invoking the function, simply add the function call
                # message back to the chat history so the agent will process it on the next round.
                if content_dict.get("function_call", {}).get("name") == "ExecuteCode":
                    arguments = content_dict["function_call"].get("arguments", {})
                    generated_code = arguments.get("generated_code", "")
                    stdout, stderr, pdf_path = run_generated_code(generated_code)
                    print(f"Execution Output:\n{stdout}")
                    if stderr:
                        print(f"Execution Error:\n{stderr}")
                    print(f"PDF saved at: {pdf_path}")
                chat.add_message(message.content)
                print("Function call received; re-invoking agent with updated history...")
                break  # Exit the inner loop to re-invoke the agent
            elif "generated_code" in content_dict:
                # If the message contains generated code, execute it.
                generated_code = content_dict["generated_code"]
                stdout, stderr, pdf_path = run_generated_code(generated_code)
                print(f"Execution Output:\n{stdout}")
                if stderr:
                    print(f"Execution Error:\n{stderr}")
                print(f"PDF saved at: {pdf_path}")
            else:
                # No function call found: assume final answer reached.
                print(message.content)
                break

        # Check the most recent message: if it is a function call, loop again to re-invoke the agent.
        last_message = chat.messages[-1]
        try:
            parsed = json.loads(last_message.content)
            if "function_call" in parsed:
                continue  # Re-invoke the agent with the updated chat history.
        except Exception:
            # If parsing fails, assume final answer is present.
            break
        break

if __name__ == "__main__":
    load_dotenv()
    csv_data = """
feature,target,col3,col4
1,2,11,0.5
2,4,12,1.0
3,6,13,1.5
4,8,14,2.0
5,10,15,2.5
6,12,16,3.0
7,14,17,3.5
8,16,18,4.0
9,18,19,4.5
10,20,20,5.0
11,22,21,5.5
12,24,22,6.0
13,26,23,6.5
14,28,24,7.0
15,30,25,7.5
16,32,26,8.0
17,34,27,8.5
18,36,28,9.0
19,38,29,9.5
20,40,30,10.0
21,42,31,10.5
22,44,32,11.0
23,46,33,11.5
24,48,34,12.0
25,50,35,12.5
26,52,36,13.0
27,54,37,13.5
28,56,38,14.0
29,58,39,14.5
30,60,40,15.0
31,62,41,15.5
32,64,42,16.0
33,66,43,16.5
34,68,44,17.0
35,70,45,17.5
36,72,46,18.0
37,74,47,18.5
38,76,48,19.0
39,78,49,19.5
40,80,50,20.0
41,82,51,20.5
42,84,52,21.0
43,86,53,21.5
44,88,54,22.0
45,90,55,22.5
46,92,56,23.0
47,94,57,23.5
48,96,58,24.0
49,98,59,24.5
50,100,60,25.0
51,102,61,25.5
52,104,62,26.0
53,106,63,26.5
54,108,64,27.0
55,110,65,27.5
56,112,66,28.0
57,114,67,28.5
58,116,68,29.0
59,118,69,29.5
60,120,70,30.0
61,122,71,30.5
62,124,72,31.0
63,126,73,31.5
64,128,74,32.0
65,130,75,32.5
66,132,76,33.0
67,134,77,33.5
68,136,78,34.0
69,138,79,34.5
70,140,80,35.0
71,142,81,35.5
72,144,82,36.0
73,146,83,36.5
74,148,84,37.0
75,150,85,37.5
76,152,86,38.0
77,154,87,38.5
78,156,88,39.0
79,158,89,39.5
80,160,90,40.0
81,162,91,40.5
82,164,92,41.0
83,166,93,41.5
84,168,94,42.0
85,170,95,42.5
86,172,96,43.0
87,174,97,43.5
88,176,98,44.0
89,178,99,44.5
90,180,100,45.0
91,182,101,45.5
92,184,102,46.0
93,186,103,46.5
94,188,104,47.0
95,190,105,47.5
96,192,106,48.0
97,194,107,48.5
98,196,108,49.0
99,198,109,49.5
100,200,110,50.0
"""
    # Define a prompt instructing the agent to generate code.
    prompt = (f"""
        You are an analysis assistant.
        You will be given a CSV data string.
        Your task is to generate Python code that reads the CSV data from a string,
        performs an analysis of this data using pandas, scipy, and scikit-learn, and outputs the results.
        The analysis should include:
            1. Reading the CSV data into a pandas DataFrame.
            2. Performing basic data exploration (e.g., summary statistics, missing values).
            3. Visualizing the data using seaborn.
            4. Performing a linear regression analysis using scikit-learn.
            5. Performing a cluster analysis using KMeans from scikit-learn.
            6. Performing a PCA analysis using scikit-learn.
            7. Perform a KS test on the data.
            8. Outputting the results of the analysis to a PDF file.
        Additionally, include a function to visualize the data using seaborn.
        
        Our response should be a JSON object representing a function call to ExecuteCode with the key generated_code.
        Do not output any plain text code.
    """)
    asyncio.run(run_code_in_docker(prompt, csv_data))
