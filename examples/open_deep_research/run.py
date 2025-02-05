import argparse
import json
import os
import threading
import time
from litellm import RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import sys
# Assuming the project root (which contains 'smolagents') lies two directories up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','src')))

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    MANAGED_AGENT_PROMPT,
    CodeAgent,
    # HfApiModel,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--run-name", type=str, required=True)
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

SET = "validation"

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

### LOAD EVALUATION DATASET

# Load your custom research questions from the CSV file
eval_df = pd.read_csv("my_research_questions.csv")

# Rename the columns to match expected names
eval_df = eval_df.rename(columns={
    "Question": "question",
    "Final answer": "true_answer",  # if you later add expected answers, leave empty if not used
    "Level": "task"
})

eval_df["file_name"].fillna("", inplace=True)

#eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
#eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})


#def preprocess_file_paths(row):
#    if len(row["file_name"]) > 0:
#        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
#    return row


#eval_df = eval_df.map(preprocess_file_paths)
#eval_df = pd.DataFrame(eval_df)
print("Loaded evaluation dataset:")
print(eval_df["task"].value_counts())

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_hierarchy(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        managed_agent_prompt=MANAGED_AGENT_PROMPT
        + """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.""",
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


def answer_single_question(example, model_id, answers_file, visual_inspection_tool):
    model = LiteLLMModel(
        model_id,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        #reasoning_effort="high",
    )
    # model = HfApiModel("Qwen/Qwen2.5-72B-Instruct", provider="together")
    #     "https://lnxyuvj02bpe6mam.us-east-1.aws.endpoints.huggingface.cloud",
    #     custom_role_conversions=custom_role_conversions,
    #     # provider="sambanova",
    #     max_tokens=8096,
    # )
    document_inspection_tool = TextInspectorTool(model, 100000)

    agent = create_agent_hierarchy(model)

    augmented_question = """You have one question to answer. Please provide your best and most accurate answer 
    using all available tools. If needed, run any verification steps to ensure the response is correct. Here 
    is the task: """ + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example.get("task_id", example.get("task", "default-task-id")),
        "true_answer": example["true_answer"],
    }

    messages = [{"role": "user", "content": augmented_question}]
    max_retries = 5
    base_wait = 2.0  # seconds
    for attempt in range(max_retries):
        try:
            response = model(messages)
            
            if hasattr(response, "content"): serialized_response = response.content 
            elif hasattr(response, "to_dict"): serialized_response = response.to_dict() 
            else: serialized_response = str(response)
            annotated_example["response"] = serialized_response
            break

        except RateLimitError as e:
            error_detail = e.args[0] if e.args else {}
            suggested_wait = error_detail.get("retry_after") if isinstance(error_detail, dict) else None
            wait_time = suggested_wait if suggested_wait is not None else base_wait * (2 ** attempt)
            print(f"Rate limit hit. Waiting for {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
    else:
        raise Exception("Exceeded maximum retry attempts due to rate limiting.")
    
    # Optionally, incorporate 'response' into annotated_example if needed.
    annotated_example["response"] = response

    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file, eval_df) -> List[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_df.to_dict(orient="records") if line["question"] not in done_questions]


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    answers_file = f"output/{SET}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_df)

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    print("All tasks processed.")


if __name__ == "__main__":
    main()
