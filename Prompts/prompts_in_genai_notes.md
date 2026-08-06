# Prompts in Generative AI — Notes

## 1. What is a Prompt?

A **prompt** is the input text (instruction, question, or context) given to a
Generative AI model (like an LLM) to produce an output. The quality, structure,
and phrasing of a prompt directly affects the quality of the model's response.

**Prompt Engineering** is the practice of designing and refining prompts to get
the most accurate, relevant, and useful output from a Generative AI model,
without changing the model's weights.

---

## 2. Why Prompts Matter

- LLMs are trained on huge, general corpora — a prompt "steers" that general
  knowledge toward a specific task.
- The same model can behave very differently depending on how the prompt is
  phrased (tone, examples given, format requested, constraints added).
- Good prompting reduces hallucination, improves format-compliance (JSON,
  tables, etc.), and can substitute for fine-tuning in many use cases.

---

## 3. Types of Prompts

| Type | Description | Example |
|---|---|---|
| **Zero-shot** | No examples given, only an instruction | "Translate this sentence to French: ..." |
| **Few-shot** | A few input-output examples are given before the actual query | "Q: 2+2? A: 4. Q: 3+5? A:" |
| **Chain-of-Thought (CoT)** | Prompt asks the model to reason step-by-step | "Let's think step by step..." |
| **Instruction prompting** | Direct command/task description | "Summarize this article in 3 bullet points." |
| **Role prompting** | Assigns a persona/role to the model | "You are a senior Python developer..." |
| **Contextual prompting** | Extra background/context is supplied along with the query | Passing a document + a question about it |
| **Self-consistency prompting** | Multiple reasoning paths sampled, majority answer chosen | Used with temperature > 0 across multiple calls |

---

## 4. Prompt Components (General Structure)

1. **Instruction** – what task to perform
2. **Context** – background information relevant to the task
3. **Input Data** – the actual data/question to act upon
4. **Output Indicator** – desired format of the response (list, JSON, tone, etc.)

---

## 5. Key Parameters That Affect Prompt Output (Generation Controls)

These are not part of the prompt text itself, but they control *how* the model
uses the prompt to generate text. They appear in the code below as
`pipeline_kwargs`.

| Parameter | Meaning |
|---|---|
| `max_length` / `max_new_tokens` | Maximum number of tokens the model is allowed to generate |
| `temperature` | Controls randomness. Low (→0) = more deterministic/focused; High (→1) = more random/creative |
| `top_k` | Limits sampling to the top-k most probable next tokens |
| `top_p` (nucleus sampling) | Limits sampling to a cumulative probability mass `p` |
| `repetition_penalty` | Penalizes the model for repeating the same tokens |

---

## 6. Code File: `prompt_ui_static.py` — Static Prompt UI using LangChain + HuggingFace + Streamlit

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

# Set up the HuggingFace model and pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Model ID for the Hugging Face model
    task = "text-generation", # Task type for the model
    pipeline_kwargs = {   # Additional arguments for the pipeline
        "max_length": 512,  
        "temperature": 0.5  # Sampling temperature for text generation
    }
) 

model = ChatHuggingFace(llm = llm)  # Initialize the chat model

# Streamlit UI
st.header("Research Tool")
user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)       
    st.write(result.content)
```

### What this script does (overview)

This is a **static prompt UI**: the user types a *free-form prompt* into a text
box, and when they click a button, that exact prompt (no template, no
pre-defined structure) is sent straight to the LLM. It's called "static"
because the prompt UI itself has one fixed input field — there's no dynamic
templating, no dropdowns for tone/format/length, and no few-shot examples
injected. Whatever the user types is exactly what the model receives.

### Line-by-line / function-by-function explanation

#### `from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline`
Imports two classes from LangChain's HuggingFace integration package:
- **`HuggingFacePipeline`** — wraps a HuggingFace `transformers` pipeline
  (e.g., text-generation) so it can be used inside LangChain like any other LLM.
- **`ChatHuggingFace`** — wraps a `HuggingFacePipeline` (or similar LLM) so it
  behaves like a **chat model** (i.e., supports `invoke()` with chat-style
  message handling and returns a message object with `.content`), even though
  the underlying model may not be natively conversational.

#### `HuggingFacePipeline.from_model_id(...)`
A **class method (constructor helper)** that downloads/loads a model from the
HuggingFace Hub and wraps it in a `transformers` pipeline, all in one call —
you don't need to manually load the tokenizer and model separately.

Parameters used:
- **`model_id`** (`str`): The HuggingFace Hub model identifier to load.
  Here it is `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` — a small (1.1B parameter)
  open-source chat-tuned LLM, good for local/lightweight experimentation.
- **`task`** (`str`): The pipeline task type — `"text-generation"` tells
  `transformers` to build a causal-LM text-generation pipeline (as opposed to
  `"text2text-generation"`, `"summarization"`, etc.).
- **`pipeline_kwargs`** (`dict`): Extra keyword arguments forwarded directly
  into the underlying HuggingFace `pipeline(...)` call / generation config:
  - `"max_length": 512` → caps the total number of tokens (prompt + generated
    output) at 512.
  - `"temperature": 0.5` → moderate randomness. 0.5 is fairly balanced —
    not too repetitive/deterministic, not too chaotic.

This call returns an `llm` object — a LangChain-compatible LLM wrapper around
the local pipeline (this model runs **locally**, not via an API call).

#### `model = ChatHuggingFace(llm = llm)`
Wraps the raw text-generation `llm` in a **chat interface**. This is necessary
because `TinyLlama-1.1B-Chat` expects input formatted using a chat template
(system/user/assistant turns), and `ChatHuggingFace` automatically applies
that chat template under the hood before passing text to the pipeline. It also
standardizes the output as a `ChatMessage`-like object, which is why the code
later accesses `result.content` instead of a raw string.

#### `st.header("Research Tool")`
Streamlit function that renders a large header/title text ("Research Tool") at
the top of the web app.

#### `user_input = st.text_input('Enter your prompt')`
Renders a **single-line text input box** with the label `"Enter your prompt"`.
Whatever the user types is captured and stored in the Python variable
`user_input`. Streamlit re-runs the script top-to-bottom on every interaction,
so this always reflects the current value of the box.

#### `if st.button('Summarize'):`
Renders a clickable **button** labeled `"Summarize"`. `st.button()` returns
`True` only during the single script-run that happens right after the button
is clicked (otherwise `False`), so the code inside the `if` block only runs
when the user actually clicks it.

#### `result = model.invoke(user_input)`
**`invoke()`** is the standard LangChain Runnable method used to run a single
call synchronously with the given input and get back the model's output.
Here, `user_input` (the raw prompt string typed by the user) is sent directly
to `model` (the `ChatHuggingFace`-wrapped LLM). Since there is no
`PromptTemplate` involved, the user's raw text becomes the entire prompt.

#### `st.write(result.content)`
- `result` is a chat-message-like object returned by `invoke()`.
- `.content` extracts just the generated text string from that object.
- `st.write()` is Streamlit's general-purpose display function — it renders
  the string content (or any Streamlit-supported object) onto the app UI.

### Why this is called a "static" prompt UI

There is exactly **one fixed prompt field**, and no logic that dynamically
builds/edits the prompt (e.g., using `PromptTemplate.from_template(...)` with
variables, few-shot example injection, or a system-message prefix). This
contrasts with **dynamic prompting UIs**, where the app:
- Uses `PromptTemplate` with placeholders (e.g., `{topic}`, `{tone}`,
  `{length}`) filled in from multiple UI widgets (dropdowns, sliders, etc.)
- Assembles a structured prompt (instruction + context + few-shot examples +
  user input) before sending it to the model
- Lets the user configure generation parameters (temperature, max tokens)
  through the UI itself, rather than hard-coding them

### Key limitations of this static approach
- No guardrails on prompt quality — output quality is fully dependent on how
  well the user phrases their own prompt.
- No reusable template — every prompt has to be typed from scratch.
- No few-shot examples to guide style/format of the output.
- `temperature` and `max_length` are hard-coded, not adjustable from the UI.

---

## 7. Static vs Dynamic Prompting — Summary

| Aspect | Static Prompt UI (this code) | Dynamic Prompt UI |
|---|---|---|
| Prompt source | Raw free text from one input box | Built from a `PromptTemplate` + variables |
| Reusability | Low — user retypes every time | High — template reused with different inputs |
| Consistency of output format | Low | High (instruction + output indicator baked in) |
| Configurability (temperature, etc.) | Hard-coded in code | Often exposed as UI controls |
| Example use case | Quick experimentation / chat box | Production tools (e.g., "Research Tool" with structured summarization prompt) |

---

## 8. Common LangChain Functions/Classes Used in Prompting Workflows (Reference)

For context, here are related functions/classes commonly used alongside what's
in this code, useful when extending a static UI into a dynamic one:

- **`PromptTemplate.from_template("...")`** — creates a reusable prompt with
  `{variable}` placeholders.
- **`ChatPromptTemplate.from_messages([...])`** — builds multi-turn prompts
  with system/human/AI message roles.
- **`model.invoke(prompt)`** — runs the model once on a given input (used in
  this code).
- **`model.stream(prompt)`** — streams tokens back as they're generated.
- **`model.batch([...])`** — runs the model on multiple inputs at once.
- **`FewShotPromptTemplate`** — injects a set of example Q/A pairs into the
  prompt automatically.

---

## 9. Quick Glossary

- **LLM**: Large Language Model.
- **Pipeline**: A HuggingFace `transformers` abstraction bundling tokenizer +
  model + pre/post-processing for a specific task.
- **Temperature**: Randomness control for sampling the next token.
- **Invoke**: LangChain's standard method to run a Runnable once.
- **Chat model wrapper**: A wrapper that formats prompts using a chat template
  and returns structured message objects instead of raw strings.

---

## 10. What is a Prompt Template?

A **Prompt Template** is a reusable, parameterized prompt — instead of
hard-coding a full prompt string, you define a string with placeholder
variables (e.g., `{topic}`, `{style}`) and fill those placeholders in
dynamically at runtime. This is the core building block that turns a
"static" prompt UI into a "dynamic" one.

**Why use templates instead of raw f-strings?**
- Built-in **input validation** (LangChain checks that all declared
  `input_variables` are supplied before running).
- Templates can be **saved to / loaded from disk** (as JSON/YAML), so they
  can be version-controlled, shared, and reused across apps.
- They compose cleanly with LangChain's **LCEL (LangChain Expression
  Language)** pipe (`|`) syntax to build chains.
- They standardize prompt structure across a team/codebase.

### Types of Prompt Templates in LangChain

| Template Class | Purpose | Used in file |
|---|---|---|
| **`PromptTemplate`** | Single plain-text prompt with `{variables}`, for plain (non-chat) LLM calls or chat models that accept a single string | `prompt_generator.py`, `prompt_ui_dynamic.py` |
| **`ChatPromptTemplate`** | A prompt built from multiple **role-tagged messages** (system/human/AI), for chat-style models | `prompt_chat_template.py`, `prompt_placeholder.py` |
| **`MessagesPlaceholder`** | A "slot" inside a `ChatPromptTemplate` where a *list* of previously-generated messages (e.g., chat history) gets inserted at runtime | `prompt_placeholder.py` |
| **`FewShotPromptTemplate`** | Injects a set of example input/output pairs into the prompt before the actual query (mentioned earlier as a reference tool) | — |

---

## 11. Code File: `prompt_generator.py` — Creating and Saving a `PromptTemplate`

```python
from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template = 
"""You are a research assistant. Your task is to summarize the research paper titled "{paper_title}" in a {summary_style} style and {summary_length} length. Please provide a clear and concise summary of the key points, findings, and contributions of the paper.
Summary:
""",
input_variables = ["paper_title", "summary_style", "summary_length"]
)

template.save("research_summary_template.json")
```

### What this script does
This file **defines and persists** the exact prompt template that is later
loaded and used by the dynamic UI (`prompt_ui_dynamic.py`). It is run once
(offline/setup step) to generate a `research_summary_template.json` file.

### Function/parameter explanation

#### `PromptTemplate(...)`
Constructor for a plain prompt template. Key parameters:
- **`template`** (`str`): The prompt text itself, containing placeholders in
  `{curly_braces}` — here `{paper_title}`, `{summary_style}`, and
  `{summary_length}`. These will be substituted with real values later.
- **`input_variables`** (`list[str]`): Explicitly declares which placeholder
  names the template expects. LangChain uses this list to **validate** that
  you supply exactly the right variables when invoking the template — if you
  forget one or pass an extra undeclared one, it raises an error early
  instead of silently producing a broken prompt.

> Note: LangChain can also auto-infer `input_variables` from the `{}` patterns
> in the string using `PromptTemplate.from_template("...")`, but declaring
> them explicitly (as done here) is more explicit and safer.

#### `template.save("research_summary_template.json")`
Serializes the template's structure (the raw string + its input variables)
into a JSON file on disk. This is what enables the template to be
**reused across different scripts/sessions** without redefining it — it only
needs to be loaded, not rebuilt.

---

## 12. Code File: `prompt_ui_dynamic.py` — Dynamic Prompt UI using a Loaded Template

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Set up the HuggingFace model and pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs = {
        "max_length": 512,  
        "temperature": 0.5
    }
) 

# Initialize the chat model
model = ChatHuggingFace(llm = llm)

# Streamlit UI
st.header("Research Tool")

# User Inputs
paper_input = st.selectbox("Select the research paper name", [...])
style_input = st.selectbox("Select the style of summary", [...])
length_input = st.selectbox("Select the length of summary", [...])

# Load the prompt template
template = load_prompt("research_summary_template.json")

# Invoke the model with the user inputs when the button is clicked
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        "paper_title": paper_input,
        "summary_style": style_input,
        "summary_length": length_input
    })

    st.write(result.content)
```

### What this script does
This is the **dynamic version** of the earlier static prompt UI. Instead of
one free-text box, the user picks structured options from dropdowns, and
those choices are slotted into a pre-built, validated prompt template — this
guarantees every prompt sent to the model follows the same well-engineered
structure, regardless of what the user selects.

### Function-by-function explanation

#### `st.selectbox(label, options_list)`
A Streamlit widget that renders a **dropdown menu**. It returns whichever
option string the user currently has selected. Three are used here:
- `paper_input` → which research paper to summarize
- `style_input` → the summary style (Concise / Detailed / Bullet Points / etc.)
- `length_input` → desired summary length (Short / Medium / Long)

Each list includes a placeholder `"Select...."` first entry so nothing is
pre-selected by default.

#### `load_prompt("research_summary_template.json")`
Loads a previously **saved** `PromptTemplate` (created and saved by
`prompt_generator.py`) back into a live `PromptTemplate` object — meaning the
prompt structure/wording only has to be engineered once and can then be
reused by any number of apps just by loading the JSON file.

#### `chain = template | model`
This is **LCEL (LangChain Expression Language)** — the `|` (pipe) operator
composes two Runnables into a **chain**: the output of the left side
(`template`, which produces a formatted prompt) is automatically fed as the
input to the right side (`model`). This is functionally equivalent to
manually doing `model.invoke(template.invoke({...}))`, but more composable
and readable, and lets you insert additional steps (like output parsers)
into the pipe later.

#### `chain.invoke({...})`
Runs the whole chain in one call. The dictionary keys (`paper_title`,
`summary_style`, `summary_length`) must exactly match the template's
declared `input_variables`. Internally:
1. `template` fills in the placeholders → produces a formatted prompt string.
2. That formatted prompt is passed to `model`, which generates a response.
3. The final output is a chat-message object, same as in the static example.

#### `st.write(result.content)`
Same as before — displays the generated text (`.content`) in the Streamlit
app.

### Static vs Dynamic — now concretely illustrated

| | Static UI (`prompt_ui_static.py`) | Dynamic UI (`prompt_ui_dynamic.py`) |
|---|---|---|
| Prompt source | Free-text box, whatever user types | `PromptTemplate` loaded from JSON, filled with dropdown selections |
| Structure guaranteed? | No — depends entirely on user's phrasing | Yes — every prompt follows the engineered template |
| Reusable across apps? | No | Yes (`research_summary_template.json` can be loaded anywhere) |
| Chaining syntax | Manual `model.invoke(user_input)` | LCEL: `template | model` then `.invoke({...})` |

---

## 13. Code File: `prompt_chat_template.py` — Multi-Role `ChatPromptTemplate`

```python
from langchain import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistant.'),
    ('human', 'Explain in simple terms, what is {topic}?')
])

prompt = chat_template.invoke({'domain': 'science', 'topic': 'quantum physics'})

print(prompt)
```

### What this script does
Builds a **chat-style prompt template** made of multiple role-tagged turns
(rather than one flat string) and fills in its placeholders.

### Function/parameter explanation

#### `ChatPromptTemplate([...])`
Constructor that accepts a **list of `(role, template_string)` tuples**. Each
tuple defines one "turn" of the conversation with a role and its own
`{placeholders}`:
- `('system', 'You are a helpful {domain} assistant.')` → sets the AI's
  persona/behavior at the start of the conversation. `{domain}` is filled
  dynamically (e.g., "science", "history", "cooking").
- `('human', 'Explain in simple terms, what is {topic}?')` → represents the
  user's message, with `{topic}` filled in dynamically.

This produces a template that, when invoked, generates a **list of message
objects** (a `SystemMessage` + a `HumanMessage`) rather than a single string
— matching how real chat models expect their input.

#### `chat_template.invoke({'domain': 'science', 'topic': 'quantum physics'})`
Fills in all placeholders across all roles at once and returns a formatted
prompt (a `ChatPromptValue`, convertible to a list of message objects) ready
to be passed into a chat model's `invoke()`.

> Note: In current LangChain versions, `ChatPromptTemplate` should be
> imported from `langchain_core.prompts` (as in the other files) rather than
> the top-level `langchain` package.

---

## 14. Code File: `prompt_placeholder.py` — `MessagesPlaceholder` for Chat History

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
])

chat_history = []
#load chat history from a file  
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})

print(prompt)
```

### What this script does
Extends the chat template idea to support **inserting an entire
variable-length conversation history** into the middle of the prompt — not
just single-value placeholders like `{topic}`.

### Function/parameter explanation

#### `MessagesPlaceholder(variable_name = 'chat_history')`
A special slot inside a `ChatPromptTemplate` that reserves a spot for a
**list of messages** to be inserted at that exact position in the
conversation, rather than a single string value. At invocation time,
whatever list is passed under the key `'chat_history'` gets expanded
in-place as multiple messages between the system prompt and the final human
query. This is exactly how production chatbots maintain multi-turn context:
`[system prompt] → [...entire past conversation...] → [new user message]`.

#### Loading chat history from file
```python
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
```
- `open('chat_history.txt')` opens the file in read mode.
- `f.readlines()` reads the file **line by line**, returning a list of
  strings (one per line, including trailing `\n` characters).
- `.extend(...)` appends all of those lines into the `chat_history` list.

> ⚠️ **Practical note:** `readlines()` here returns raw plain-text lines, not
> actual LangChain message objects (`HumanMessage`/`AIMessage`). For this to
> work correctly with `MessagesPlaceholder` in real usage, `chat_history`
> should instead contain a list of message objects, e.g., built like:
> `[HumanMessage(content="..."), AIMessage(content="..."), ...]` — otherwise
> the placeholder is being filled with raw strings rather than proper chat
> turns.

#### `chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})`
Fills in the `MessagesPlaceholder` slot with the entire `chat_history` list,
and the `{query}` placeholder in the final human turn with the string
`'where is my refund'`. The result is a fully assembled multi-turn prompt
ready to send to a chat model.

---

## 15. Types of Messages in LangChain

When working with **chat models** (as opposed to plain text-completion
models), the conversation is represented as a sequence of typed **message
objects**, not a flat string. Each message has a `role` and `content`.

| Message Class | Role it represents | Purpose |
|---|---|---|
| **`SystemMessage`** | `system` | Sets the model's behavior, persona, or instructions for the whole conversation (usually the first message) |
| **`HumanMessage`** | `user` / `human` | Represents input typed/sent by the end user |
| **`AIMessage`** | `assistant` / `ai` | Represents a response previously generated by the model (used to preserve conversational memory) |
| **`ChatMessage`** | custom | A generic message where you specify an arbitrary role string yourself |
| **`FunctionMessage` / `ToolMessage`** | `function` / `tool` | Represents the result of a tool/function call, fed back to the model as context |

In `ChatPromptTemplate`, the same roles are expressed as simple tuples:
`('system', "...")`, `('human', "...")`, `('ai', "...")` — LangChain converts
these into the corresponding message objects internally.

Maintaining a **`chat_history` list** of `SystemMessage` / `HumanMessage` /
`AIMessage` objects (appending each new turn as it happens) is exactly how
LLMs — which are inherently stateless — are given the illusion of "memory"
across a conversation: the entire history is re-sent with every new request.

---

## 16. Code File: Chatbot #1 — Continuous Console Chatbot with Memory

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,   # <-- key fix: only return the new reply
    }
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)
```

### What this script does
Implements a **looping, terminal-based chatbot** that remembers everything
said in the current session by continuously growing a `chat_history` list of
message objects and re-sending the whole list on every turn.

### Function/parameter explanation

#### `HuggingFacePipeline.from_model_id(...)` (new parameters vs. earlier files)
- **`model_id = "Qwen/Qwen2.5-1.5B-Instruct"`**: A different, instruction-tuned
  open-source chat model from Alibaba's Qwen family.
- **`max_new_tokens: 512`**: Unlike `max_length` (used in earlier files, which
  caps *prompt + output* combined), `max_new_tokens` caps **only the newly
  generated output**, regardless of how long the input prompt/history is —
  important for a growing chat history, where the total context keeps
  increasing.
- **`do_sample: True`**: Enables **probabilistic sampling** of the next
  token (using `temperature`/`top_k`/`top_p`) instead of always greedily
  picking the single highest-probability token. Without this, `temperature`
  would effectively have no effect.
- **`return_full_text: False`**: By default, HuggingFace `text-generation`
  pipelines return the prompt **plus** the generated continuation
  concatenated together. Setting this to `False` makes the pipeline return
  **only the newly generated reply**, not the entire prompt echoed back —
  necessary here, otherwise `result.content` would contain the whole chat
  history again inside the "reply."

#### Building and updating `chat_history`
- Starts as a list containing just one `SystemMessage` — establishing the
  assistant's behavior before any user turns happen.
- **`input("You: ")`**: Python's built-in function to read a line of text
  typed by the user in the terminal.
- `if user_input == "exit": break`: Lets the user type `"exit"` to end the
  loop and stop the chatbot.
- **`chat_history.append(HumanMessage(content=user_input))`**: Records the
  user's new message as part of the conversation history.
- **`model.invoke(chat_history)`**: Sends the **entire accumulated
  conversation** (system + all past turns) to the model in one call — this is
  what gives the chatbot "memory" of earlier turns, since the model itself
  has no built-in memory between calls.
- **`chat_history.append(AIMessage(content=result.content))`**: Records the
  model's reply back into history, so it's included as context in the *next*
  turn too.
- The `while True:` loop repeats this indefinitely until `"exit"` is typed.
- `print(chat_history)` at the end dumps the full list of message objects
  once the loop breaks — useful for debugging/inspection.

---

## 17. Code File: Chatbot #2 — Single-Turn Message List (No Loop)

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
        "return_full_text": False,
    }
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello! How are you?"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
```

### What this script does
This is essentially the **minimal, non-looping building block** that
Chatbot #1's `while` loop is built on top of — it demonstrates a single
request/response exchange using explicit message objects instead of a raw
string, without any interactive loop.

### Function/parameter explanation
- **`messages = [SystemMessage(...), HumanMessage(...)]`**: Manually
  constructs a two-turn conversation: a system instruction followed by one
  fixed human question ("Hello! How are you?") — no `input()` call, so the
  question is hard-coded rather than typed by a user.
- **`model.invoke(messages)`**: Sends this list of message objects to the
  chat model in one call, exactly the same mechanism used inside the loop in
  Chatbot #1.
- **`messages.append(AIMessage(content=result.content))`**: Adds the model's
  reply into the `messages` list — demonstrating how you'd manually extend
  history for a *next* turn, even though this script doesn't actually make a
  second call.
- **`print(messages)`**: Prints the full list (system + human + AI message
  objects) to inspect the final conversation state.

### Chatbot #1 vs Chatbot #2 — Comparison

| | Chatbot #1 (looping) | Chatbot #2 (single-turn) |
|---|---|---|
| Interaction | Continuous `while True` loop reading `input()` | One hard-coded exchange, no loop |
| User input source | Live terminal input | Hard-coded string in code |
| History growth | Grows every iteration | Grows once, manually, after the single call |
| Use case | An actual usable interactive chatbot | A minimal demo of message-based `invoke()` |

---

## 18. `max_length` vs `max_new_tokens` — Important Distinction

| Parameter | Counts | Risk if too small with growing chat history |
|---|---|---|
| `max_length` | Prompt tokens **+** generated tokens combined | As chat history grows, less and less room is left for the actual new reply — can silently truncate or fail |
| `max_new_tokens` | **Only** the newly generated tokens | Reply length stays consistent regardless of how long the input history gets |

This is exactly why the static/dynamic UI files (single-turn, short prompts)
use `max_length`, while the chatbot files (accumulating history) switch to
`max_new_tokens`.

---

## 19. Full File Summary Table

| # | File | Prompt style | Key concept demonstrated |
|---|---|---|---|
| 1 | `prompt_ui_static.py` | Raw free-text prompt | Static prompting, basic `invoke()` |
| 2 | `prompt_ui_dynamic.py` | Loaded `PromptTemplate` + dropdowns | Dynamic prompting, LCEL chains (`|`) |
| 3 | `prompt_generator.py` | `PromptTemplate` definition | Creating & saving reusable templates |
| 4 | `prompt_placeholder.py` | `ChatPromptTemplate` + `MessagesPlaceholder` | Injecting chat history into a template |
| 5 | `prompt_chat_template.py` | `ChatPromptTemplate` | Multi-role prompt templates |
| 6 | Chatbot #1 | List of message objects, looped | Stateful multi-turn chatbot |
| 7 | Chatbot #2 | List of message objects, single call | Minimal message-based `invoke()` |
