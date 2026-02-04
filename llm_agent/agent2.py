'''
Simple LLM agent (uses OLlama / Phi4:14b model)
   Agent capabilities: reads, writes, and edit files (asks permission to write/modify files), searches web

Install phi4:14b model (via ollama)
ollama pull phi4:14b

Install duckduckgo web search library:
pip install -U ddgs

This vs. production tools like Claude, ChatGPT
    Better error handling and fallback behaviors
    Streaming responses for better UX
    Smarter context management (summarizing long files, etc.)
    More tools (run commands, search codebase, etc.)
    Approval workflows for destructive operations

    Other tool ideas: add thinking, summarization

    Coding specific functions:
       suggest code, explain errors, write functions, 
'''

import inspect 
import json
import ollama

from pathlib import Path
from ddgs import DDGS
from json.decoder import JSONDecodeError
from typing import Any, Dict, List#, Tuple

# Agent Configuration
MODEL = "phi4:14b"

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"

# Web search configuration
MAX_WEB_RESULTS = 4
DEBUG_TO_CONSOLE = False

def find_json_attribute_index(attribute_name: str, text: str) -> int:
    """ Just a convenience method for finding index of a attribute
    """

    if f'"{attribute_name}": "' in text:
        return text.index(f'"{attribute_name}": "')

    if f"'{attribute_name}': '" in text:
        return text.index(f"'{attribute_name}': '")
    
    return -1

def resolve_abs_path(path_str: str) -> Path:
    """
    Resolves local file path to absolute path.
    Example: file.py -> /Users/you/project/file.py
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path

# --- Agent Tools must provide desctiption of the function, input parameters, and return
def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: Dictionary that contains: file path, and the full content of the file.
    """
    full_path = resolve_abs_path(filename)
    try:
        with open(str(full_path), "r") as f:
            return {"file_path": str(full_path), "content": f.read()}
    except Exception as e:
        return {"error": str(e)}


def list_files_tool(path: str) -> Dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({
            "filename": item.name,
            "type": "file" if item.is_file() else "dir"
        })
    return {
        "path": str(full_path),
        "files": all_files
    }

def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file and the action taken.
    """
    full_path = resolve_abs_path(path)

    #confirm file change
    confirm = input(f"{ASSISTANT_COLOR}Do you want to write/overwrite '{full_path}'? {RESET_COLOR} {YOU_COLOR}Your response (yes/no):{RESET_COLOR} ").strip().lower()
    if not (confirm == 'yes' or confirm == 'y'):
        return {"status": "cancelled", "file_path": None, "message": "Operation cancelled by the user."}

    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created_file"
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {
            "path": str(full_path),
            "action": "old_str not found"
        }
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {
        "path": str(full_path),
        "action": "edited"
    }


def write_file_tool(filename: str, content: str) -> Dict[str, Any]:
    """
    Creates a new file named filename with content,
    create/overwrite file with new_str.
    :param filename: The file name to create.
    :param content: The contents of the file.
    :return: A dictionary with the path to the file and the action taken.
    """
    full_path = resolve_abs_path(filename)

    confirm = input(f"{ASSISTANT_COLOR}Do you want to write/overwrite '{full_path}'? {RESET_COLOR} {YOU_COLOR}Your response (yes/no):{RESET_COLOR} ").strip().lower()
    if not (confirm == 'yes' or confirm == 'y'):
        return {"status": "cancelled", "file_path": None, "message": "Operation cancelled by the user."}

    try:
        with open(str(full_path), "w", encoding="utf-8") as f:
            f.write(str(content))
        return {"status": "success", "file_path": str(full_path)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def web_search_tool(query: str)-> List[Dict]:
    """
    Search the web using DuckDuckGo Search API and return top results. Use web search to find current information.
    :param  query: The search query string.
    :return: A list of dictionaries containing relevant information from search results.
    """
    if DEBUG_TO_CONSOLE:
        print(f"DEBUG: web_search_tool(query={query})")

    results = [{"title": "", "href": "", "body": ""}]  # fallback in case of an error
    result_header = ""
    
    try:
        # Execute the search using DDGS; returns a list of dictionaries
        results = DDGS().text(query, max_results=MAX_WEB_RESULTS)

        # Start the result header in YAML format
        result_header = "```yaml\nweb_search_query: \"" + query + "\"\nsearch_results:\n"

        # Format each result into a YAML block with consistent indentation
        for r in results:
            result_entry = (
                f"  - title: \"{r['title']}\"\n"
                f"    url: {r['href']}\n"
                "    body: |\n"  # Use '|' to ensure literal block style for multiline content
            )
            
            # Add the body, ensuring proper indentation (two spaces more than other keys)
            indented_body = "\n".join(
                ["      " + line if i > 0 else "      " + line.strip()
                 for i, line in enumerate(r['body'].replace('"', "'").splitlines())]
            )
            result_entry += f"{indented_body}\n"
            result_header += result_entry

        result_header += "\n```"

    except Exception as e:
        result_header = "Web search failed. No connection."
        print("Warning:", str(e))

    if DEBUG_TO_CONSOLE:
        print(f"\n\nDEBUG: web_search_tool() - result_header:\n{result_header}")

    return result_header

# The LLM needs to know what tools exist and how to call them.
# We generate this dynamically from our function signatures and docstrings:
# TOOL REGISTRY - Mapping tools for easy execution
TOOL_REGISTRY = {
    "list_files": list_files_tool,
    "read_file": read_file_tool,
    "edit_file": edit_file_tool,
    "write_file": write_file_tool,
    "search_web": web_search_tool
}

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks. 
You have access to a series of tools you can execute. Here are the tools you can execute:

{tool_list_repr}

When you want to use a tool, reply with exactly the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
"""

def get_tool_str_representation(tool_name: str) -> str:
    ''' This function is used to get the string representation of each agent tool.'''
    tool = TOOL_REGISTRY[tool_name]
    return f"""
    Name: {tool_name}
    Description: {tool.__doc__}
    Signature: {inspect.signature(tool)}
    """

def get_full_system_prompt():
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
        tool_str_repr += f"\n{'='*15}\n"
    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr)


def execute_llm_call(conversation: List[Dict[str, str]]):
    # Re-assemble to ensure system is first, then users/assistants
    system_msgs = [m for m in conversation if m["role"] == "system"]
    other_msgs = [m for m in conversation if m["role"] != "system"]
    
    # Combine them (System first is best practice for local models)
    formatted_messages = system_msgs + other_msgs

    response = ollama.chat(
        model="phi4:14b",
        messages=formatted_messages,
    )
    
    return response['message']['content']


def chat_loop():
    messages = [{"role": "system", "content": get_full_system_prompt()}]
    
    while True:
        user_input = input(f"{YOU_COLOR}You: {RESET_COLOR}")
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]: break
        
        messages.append({"role": "user", "content": user_input})
        
        # Inner loop to handle multiple tool calls if necessary
        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
            )
            
            response_text = response['message']['content'].strip()
            messages.append({"role": "assistant", "content": response_text})

            result = ""
            
            #=======================
            #  Check for tool call
            #=======================

            # no tool call, just interaction with agent
            if 'tool:' not in response_text:
                print(f"{ASSISTANT_COLOR}Assistant: {response_text}{RESET_COLOR}")
                break
            
            #responce contains tool action
            else:

                #initialize local variables
                description = ""
                tool_action = ""
                tool_idx = response_text.index('tool:')

                # check if responce contains description (then followed by a tool action)
                if tool_idx > 0:
                    # split response description from tool action
                    description = response_text[:tool_idx-1]
                    tool_action = response_text[tool_idx+5:]
                    print(f"{ASSISTANT_COLOR}{description}{RESET_COLOR}")
                
                # tool action is the only call, strips the 'tool:' from the tool call
                else:
                    tool_action = response_text[tool_idx+5:]

                #Process tool action
                print(f"{ASSISTANT_COLOR}[*] Executing {tool_action[:30]}...{RESET_COLOR}")

                # 1. strip tool command out
                command = tool_action[:tool_action.index('(')].strip()

                # 2. strip the parameter data for the tool as raw jason
                raw_json = tool_action[tool_action.index('{'):tool_action.rfind('}')+1]

                # 3. Generated JSON often contain unescaped double qoutes, manually handle the JSON content element
                if command == 'write_file':
                    idx1 = find_json_attribute_index('content', raw_json)
                    idx2 = raw_json.rfind('}')
                    file_content = raw_json[idx1+12:idx2-1]

                    #temporarily set content to empty, then create a dictionary,
                    raw_json = raw_json.replace(file_content, "")
                    args = json.loads(raw_json)

                    # the replace otherwise the contents of the file becomes flat.
                    args['content'] = file_content.replace(r'\n', '\n')

                    # print(f"\nRAW_JSON: {raw_json}\nCMD: {command}\nCNT: {file_content}")
                    print(f"\nCMD: {command}\nCNT: type={type(file_content)}, val={file_content}")
                    print(f"[*] Executing {command}...")
                    result = TOOL_REGISTRY[command](**args)

                # Technically the 3 agent tool action calls below use identical code.
                # However, these are seperated for each agent action in case one needs to do additional processing. 
                elif command == 'read_file':
                    #attributes: filename
                    args = json.loads(raw_json)
                    print(f"[*] Executing {command}...")
                    result = TOOL_REGISTRY[command](**args)

                elif command == 'list_files':
                    #attributes: path
                    args = json.loads(raw_json)
                    print(f"[*] Executing {command}...")
                    result = TOOL_REGISTRY[command](**args)

                elif command == 'edit_file':
                    #attributes: path, old_str, new_str
                    args = json.loads(raw_json)
                    print(f"[*] Executing {command}...")
                    result = TOOL_REGISTRY[command](**args)

                elif command == 'search_web':
                    #attributes: query
                    args = json.loads(raw_json)
                    print(f"[*] Executing {command}...")
                    result = TOOL_REGISTRY[command](**args)
                    
                #update LLM state with the action taken
                messages.append({
                    "role": "user", 
                    "content": f"tool_result: {json.dumps(result)}"
                })
                continue # Let the model process the result
                

if __name__ == "__main__":
    print("=== Agent-2 (File access, web search, )===")
    chat_loop()

#   create me an ascii art of a cat and save it as cat.txt
#   show me the contents of the cat.txt file
#   edit the cat.txt file and replace o with x
#   create me a python code that prints first 10 fibinacci sequence. save it as fib.py