import asyncio
from app.agents import ToolerOrchestrator
from app.schemas import Assembly, Agent


def create_sample_assembly() -> Assembly:
    """
    Create a sample assembly with one agent configured for code generation.
    The agent (with objective 'code') is expected to generate Python analysis code
    that will be executed in a secure Docker container via CodeRunnerPlugin.
    """

    sample_agent = Agent(
        id="agent1",
        name="CodeGenAgent",
        metaprompt=(
            "You are a Python code generation agent. Generate code that loads a CSV file, "
            "which is mounted as 'uploaded_file.csv', and prints its first five rows. "
            "Do not include any additional commentary – just produce valid Python code."
        ),
        model_id="default",
        objective="code"
    )

    return Assembly(
        id="sample_assembly",
        agents=[sample_agent],
        objective="CSV Analysis Code Generation",
        roles=["code_generator"],
    )

async def main():
    # Create the sample assembly that defines our agent(s).
    assembly = create_sample_assembly()
    prompt = """
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
    orchestrator = ToolerOrchestrator()
    print("Aggregated Responses:")
    response = await orchestrator.run_interaction(assembly, prompt, strategy="llm")
    print(len(response), '\n')
    for item in response:
        for subitem in item:
            print(subitem, '\n\n')

if __name__ == '__main__':
    asyncio.run(main())
