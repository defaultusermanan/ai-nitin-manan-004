#!/usr/bin/env python3

# streamlit_github_ui_fixed_codeql.py

"""

Streamlit UI to build and send github-mcp-server JSON payloads.

Features:

 - Build create_repository / create_branch / create_or_update_file / multi_file / create_issue payloads

 - Accept pasted LLM multi_file JSON or produce multi_file from sandboxed workspace

 - Create CodeQL workflow from local canonical template

 - Generate repository docs via Azure OpenAI (scans repo), produce Markdown pages

 - Publish generated wiki pages to GitHub Wiki by cloning <owner>/<repo>.wiki.git, committing and pushing

 - Option: send file contents to MCP proxy as plain text (default) or base64 (for compatibility)

"""

import os

import re

import json

import base64

import textwrap

import tempfile

import shutil

import subprocess

from contextlib import contextmanager

from typing import List, Dict, Tuple

import requests

import streamlit as st

# --- Config / env ---

st.set_page_config(page_title="MCP GitHub Prompt UI (CodeQL + Wiki push)", layout="wide")

DEFAULT_MCP_PROXY = os.environ.get("MCP_PROXY", "http://127.0.0.1:8080/call")

GITHUB_PAT = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") # required for scanning/pushing wiki

AZ_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

AZ_KEY = os.environ.get("AZURE_OPENAI_KEY")

AZ_DEPLOY = os.environ.get("AZURE_DEPLOYMENT")

AZ_API_VER = os.environ.get("AZURE_API_VERSION", "2023-05-15")

# --- Canonical CodeQL workflow template ---

CODEQL_WORKFLOW_TEMPLATE = textwrap.dedent("""

name: "CodeQL"

on:

 push:

  branches: [ "{branch}" ]

 pull_request:

  branches: [ "{branch}" ]

 schedule:

  - cron: '0 3 * * 0' # weekly

jobs:

 analyze:

  name: Analyze

  runs-on: ubuntu-latest

  permissions:

   actions: read

   contents: read

   security-events: write

  strategy:

   fail-fast: false

   matrix:

    language: [ 'javascript', 'typescript' ]

  steps:

   - name: Checkout repository

    uses: actions/checkout@v4

   - name: Initialize CodeQL

    uses: github/codeql-action/init@v3

    with:

     languages: ${{ matrix.language }}

   - name: Install dependencies

    run: |

     if [ -f package-lock.json ]; then

      npm ci

     elif [ -f yarn.lock ]; then

      yarn install

     else

      npm install

     fi

   - name: Autobuild

    uses: github/codeql-action/autobuild@v3

   - name: Perform CodeQL Analysis

    uses: github/codeql-action/analyze@v3

""").strip()

# --- Helpers ---

def b64_of_text(text: str) -> str:

  return base64.b64encode(text.encode("utf-8")).decode("ascii")

def is_base64(s: str) -> bool:

  if not isinstance(s, str) or not s:

    return False

  try:

    base64.b64decode(s, validate=True)

    return True

  except Exception:

    return False

def strip_code_fence(s: str) -> str:

  s = s.strip()

  if s.startswith("```") and s.endswith("```"):

    parts = s.split("\n")

    if parts[0].startswith("```"):

      return "\n".join(parts[1:-1]).strip()

  return s.strip("` \n\r\t")

def parse_json_from_model(text: str) -> dict:

  t = strip_code_fence(text)

  try:

    return json.loads(t)

  except Exception as e:

    first = t.find("{")

    last = t.rfind("}")

    if first != -1 and last != -1 and last > first:

      try:

        return json.loads(t[first:last+1])

      except Exception:

        pass

    raise ValueError(f"Failed to parse JSON from model output: {e}\nRaw output:\n{t}")

@contextmanager

def ephemeral_workspace():

  td = tempfile.mkdtemp(prefix="mcp_ephemeral_")

  try:

    yield td

  finally:

    try:

      shutil.rmtree(td)

    except Exception:

      pass

def write_files_to_dir(files: List[Dict], destdir: str):

  for f in files:

    path = f.get("path")

    if not path:

      continue

    full = os.path.join(destdir, path)

    os.makedirs(os.path.dirname(full) or full, exist_ok=True)

    content = f.get("content")

    if content is None and "local_path" in f:

      shutil.copyfile(f["local_path"], full)

      continue

    if isinstance(content, str) and is_base64(content):

      data = base64.b64decode(content)

      with open(full, "wb") as fh:

        fh.write(data)

    elif isinstance(content, str):

      with open(full, "w", encoding="utf-8") as fh:

        fh.write(content)

    else:

      with open(full, "wb") as fh:

        fh.write(content)

def package_dir_to_files(dirpath: str) -> List[Dict]:

  files = []

  for root, _, filenames in os.walk(dirpath):

    for fn in filenames:

      fp = os.path.join(root, fn)

      rel = os.path.relpath(fp, dirpath).replace("\\", "/")

      with open(fp, "rb") as fh:

        b = fh.read()

      files.append({"path": rel, "content": base64.b64encode(b).decode("ascii")})

  return files

def run_commands_in_docker(workdir: str, image: str, commands: List[str], timeout_seconds: int = 120, allow_network: bool = False):

  try:

    subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  except Exception as e:

    return False, f"docker not available: {e}", 1

  logs = []

  for cmd in commands:

    net_args = []

    if not allow_network:

      net_args = ["--network", "none"]

    docker_cmd = [

      "docker", "run", "--rm",

      "-v", f"{os.path.abspath(workdir)}:/work",

      "-w", "/work"

    ] + net_args + [image, "sh", "-lc", cmd]

    try:

      proc = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_seconds)

      out = proc.stdout.decode("utf-8", errors="ignore")

      logs.append(f"$ {cmd}\n{out}")

      if proc.returncode != 0:

        return False, "\n\n".join(logs), proc.returncode

    except subprocess.TimeoutExpired:

      return False, "\n\n".join(logs) + f"\nCommand timed out after {timeout_seconds}s", 124

    except Exception as e:

      return False, "\n\n".join(logs) + f"\nException: {e}", 1

  return True, "\n\n".join(logs), 0

# --- Azure OpenAI helper ---

def azure_chat_completion(user_text: str, temperature: float = 0.0, max_tokens: int = 800) -> str:

  if not (AZ_ENDPOINT and AZ_KEY and AZ_DEPLOY):

    raise RuntimeError("Azure config missing. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_DEPLOYMENT")

  url = f"{AZ_ENDPOINT.rstrip('/')}/openai/deployments/{AZ_DEPLOY}/chat/completions?api-version={AZ_API_VER}"

  # optional local protocol KB

  kb_path = os.path.join(os.path.dirname(__file__), "github_mcp_protocol.md")

  protocol_kb = ""

  if os.path.exists(kb_path):

    try:

      with open(kb_path, "r", encoding="utf-8") as f:

        protocol_kb = f.read().strip()

    except Exception:

      protocol_kb = ""

  base_system = (

    "You are a documentation writer assistant. ALWAYS output valid JSON only for the requested format. "

  )

  system_prompt = (protocol_kb + "\n\n" + base_system) if protocol_kb else base_system

  body = {

    "messages": [

      {"role": "system", "content": system_prompt},

      {"role": "user", "content": user_text}

    ],

    "max_tokens": max_tokens,

    "temperature": temperature,

    "top_p": 1.0,

    "n": 1

  }

  headers = {"Content-Type": "application/json", "api-key": AZ_KEY}

  r = requests.post(url, headers=headers, json=body, timeout=120)

  r.raise_for_status()

  j = r.json()

  return j["choices"][0]["message"]["content"]

# --- GitHub helpers (scan repo, fetch file) ---

def fetch_repo_tree(owner: str, repo: str, ref: str = "main") -> dict:

  if not GITHUB_PAT:

    raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN not set in environment (required to scan repo).")

  url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"

  headers = {"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}

  r = requests.get(url, headers=headers, timeout=30)

  r.raise_for_status()

  return r.json()

def fetch_file_content(owner: str, repo: str, path: str, ref: str = "main") -> str:

  if not GITHUB_PAT:

    raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN not set in environment (required to fetch files).")

  url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

  headers = {"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}

  params = {"ref": ref}

  r = requests.get(url, headers=headers, params=params, timeout=30)

  r.raise_for_status()

  j = r.json()

  if j.get("encoding") == "base64" and "content" in j:

    return base64.b64decode(j["content"]).decode("utf-8", errors="ignore")

  return j.get("content", "")

# --- Function to publish wiki by cloning/pushing wiki.git ---

def push_files_to_wiki_git(owner: str, repo: str, files: List[Dict], commit_message: str = "Add wiki pages (MCP UI)") -> Tuple[bool, str]:

  """

  Clone the <owner>/<repo>.wiki.git using an HTTPS URL that embeds the PAT, write files, commit and push.

  Returns (ok, logs). Does not print the PAT anywhere in logs.

  files: [{"path": "Home.md", "content": "markdown content (plain text)"}]

  """

  token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or GITHUB_PAT

  if not token:

    return False, "GITHUB_PERSONAL_ACCESS_TOKEN not set in environment"

  tempdir = tempfile.mkdtemp(prefix="mcp_wiki_")

  logs = []

  try:

    # Construct authenticated clone URL (token embedded). We will NOT print the token.

    auth_remote = f"https://{token}@github.com/{owner}/{repo}.wiki.git"

    normal_remote = f"https://github.com/{owner}/{repo}.wiki.git"

    # For logs, do not show the token; show a masked placeholder instead

    logs.append(f"Cloning wiki remote: {normal_remote} (using PAT provided in environment)")

    # Clone

    clone_proc = subprocess.run(

      ["git", "clone", auth_remote, tempdir],

      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180

    )

    clone_out = clone_proc.stdout.decode("utf-8", errors="ignore")

    logs.append("git clone output:\n" + clone_out)

    if clone_proc.returncode != 0:

      return False, "\n".join(logs)

    # Configure a safe local git user (so commits succeed)

    try:

      subprocess.run(["git", "-C", tempdir, "config", "user.email", "mcp-ui@example.com"], check=True,

              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      subprocess.run(["git", "-C", tempdir, "config", "user.name", "MCP UI"], check=True,

              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      logs.append("Set local git user config")

    except Exception as e:

      logs.append(f"Warning: failed to set git config: {e}")

    # Write files into the cloned wiki working tree (overwrite if present)

    for f in files:

      path = f.get("path")

      content = f.get("content", "")

      if not path:

        continue

      full = os.path.join(tempdir, path)

      os.makedirs(os.path.dirname(full) or full, exist_ok=True)

      # content should be plain text here (wiki expects markdown). If it's base64, decode.

      if isinstance(content, str) and is_base64(content):

        try:

          text = base64.b64decode(content).decode("utf-8")

        except Exception:

          text = content

      else:

        text = content

      with open(full, "w", encoding="utf-8") as fh:

        fh.write(text)

      logs.append(f"Wrote wiki file: {path}")

    # git add, commit

    add_proc = subprocess.run(["git", "-C", tempdir, "add", "--all"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    logs.append("git add output:\n" + add_proc.stdout.decode("utf-8", errors="ignore"))

    commit_proc = subprocess.run(["git", "-C", tempdir, "commit", "-m", commit_message],

                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    commit_out = commit_proc.stdout.decode("utf-8", errors="ignore")

    logs.append("git commit output:\n" + commit_out)

    # if nothing to commit, it's okay — continue

    if commit_proc.returncode != 0:

      if "nothing to commit" in commit_out.lower() or "no changes added to commit" in commit_out.lower():

        logs.append("No changes to commit (nothing new).")

      else:

        return False, "\n".join(logs)

    # push using the authenticated remote (origin currently points to the URL with token)

    push_proc = subprocess.run(["git", "-C", tempdir, "push", "origin", "HEAD"],

                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)

    push_out = push_proc.stdout.decode("utf-8", errors="ignore")

    logs.append("git push output:\n" + push_out)

    if push_proc.returncode != 0:

      return False, "\n".join(logs)

    # For hygiene: reset the remote URL to the normal non-token variant to avoid keeping token in config

    try:

      subprocess.run(["git", "-C", tempdir, "remote", "set-url", "origin", normal_remote],

              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      logs.append("Reset remote URL to token-less URL for hygiene.")

    except Exception as e:

      logs.append(f"Warning: failed to reset remote URL: {e}")

    return True, "\n".join(logs)

  except subprocess.TimeoutExpired:

    return False, "\n".join(logs) + "\nTimeout during git operation"

  except Exception as e:

    logs.append(f"Exception: {e}")

    return False, "\n".join(logs)

  finally:

    try:

      shutil.rmtree(tempdir)

    except Exception:

      pass
# --- UI Layout ---

st.title("MCP — GitHub Prompt UI (CodeQL + Wiki auto-publish)")

with st.sidebar:

  st.header("Settings")

  mcp_url = st.text_input("MCP proxy URL", value=DEFAULT_MCP_PROXY)

  st.markdown("Make sure your http-proxy is running and points to the stdio MCP server.")

  st.markdown("Environment checks:")

  st.text("GITHUB_PAT: " + ("set" if GITHUB_PAT else "NOT SET"))

  st.text("git available: " + ("yes" if shutil.which("git") else "no"))

  st.text("Azure config: " + ("set" if (AZ_ENDPOINT and AZ_KEY and AZ_DEPLOY) else "NOT SET"))

  st.markdown("---")

  st.checkbox("Send file content to MCP proxy base64-encoded (compat mode)", value=False, key="send_base64_default")

st.markdown("Write a natural prompt (optional):")

prompt_text = st.text_area("Natural prompt (optional)", value="", height=120,

             placeholder="e.g. Create a repo named node-demo, add branch feature-auth and add CodeQL workflow")

st.write("Select operations to perform:")

col1, col2 = st.columns(2)

with col1:

  op_repo = st.checkbox("Create repository", value=True)

  op_branch = st.checkbox("Create branch", value=True)

with col2:

  op_codeql = st.checkbox("Create CodeQL workflow", value=True)

  op_issue = st.checkbox("Create issue", value=False)

st.markdown("### Details (fill for reliability)")

owner = st.text_input("Owner (GitHub username/org)", value=os.environ.get("GITHUB_OWNER", "defaultusermanan"))

repo_name = st.text_input("Repository name (leave blank to infer)", value="")

branch_name = st.text_input("Branch name (leave blank to infer / default 'main')", value="")

issue_title = st.text_input("Issue title", value="Please review CodeQL setup")

issue_body = st.text_area("Issue body (optional)", value="Please verify CodeQL results and any alerts.")

send_as_base64 = st.checkbox("Send file contents as base64 to MCP (override sidebar)", value=False)

# LLM multi_file paste area

st.markdown("### Optional: paste LLM `multi_file` JSON (or model output) here")

llm_json_text = st.text_area("LLM JSON (paste full JSON returned by model)", value="", height=200)

process_llm_btn = st.button("Process LLM payload")

# Build/Send buttons

btn_build, btn_send = st.columns(2)

with btn_build:

  build_pressed = st.button("Build payloads")

with btn_send:

  send_pressed = st.button("Send to MCP proxy")

if "built_payloads" not in st.session_state:

  st.session_state.built_payloads = []

if "wiki_payloads" not in st.session_state:

  st.session_state.wiki_payloads = []

# --- Helpers to infer names from prompt ---

def infer_from_prompt(prompt: str):

  repo = ""

  branch = ""

  if not prompt:

    return repo, branch

  m = re.search(r"(?:repo|repository|named|called)\s+(?:['\"])?([A-Za-z0-9._-]+)(?:['\"])?", prompt, re.IGNORECASE)

  if m:

    repo = m.group(1)

  m2 = re.search(r"branch\s+(?:named\s+)?(?:['\"])?([A-Za-z0-9._/-]+)(?:['\"])?", prompt, re.IGNORECASE)

  if m2:

    branch = m2.group(1)

  return repo, branch

# --- Build payloads ---

if build_pressed:

  inferred_repo, inferred_branch = infer_from_prompt(prompt_text)

  target_repo = (repo_name.strip() or inferred_repo or "").strip()

  target_branch_for_workflow = (branch_name.strip() or inferred_branch or "main").strip()

  st.session_state.built_payloads = []

  errors = []

  if not any([op_repo, op_branch, op_codeql, op_issue]) and not llm_json_text.strip():

    st.error("Please select at least one operation or paste an LLM multi_file JSON.")

  else:

    if op_repo:

      if not target_repo:

        errors.append("Repository name is required to create a repository (fill the field or include in prompt).")

      else:

        st.session_state.built_payloads.append({

          "tool": "create_repository",

          "arguments": {

            "name": target_repo,

            "description": f"Repository created via MCP UI (prompt: {prompt_text[:140]})",

            "private": False,

            "autoInit": True

          }

        })

    if op_branch:

      if not target_repo:

        errors.append("Repository name is required to create a branch.")

      else:

        st.session_state.built_payloads.append({

          "tool": "create_branch",

          "arguments": {

            "owner": owner,

            "repo": target_repo,

            "new_branch": (branch_name.strip() or inferred_branch or "feature-branch"),

            "base_branch": "main"

          }

        })

    if op_codeql:

      if not target_repo:

        errors.append("Repository name is required to add a CodeQL workflow file.")

      else:

        codeql_content = CODEQL_WORKFLOW_TEMPLATE.format(branch=target_branch_for_workflow)

        content_payload = b64_of_text(codeql_content) if (send_as_base64 or st.session_state.get("send_base64_default")) else codeql_content

        st.session_state.built_payloads.append({

          "tool": "create_or_update_file",

          "arguments": {

            "owner": owner,

            "repo": target_repo,

            "branch": "main",

            "path": ".github/workflows/codeql.yml",

            "message": "Add CodeQL workflow (canonical)",

            "content": content_payload

          }

        })

    if op_issue:

      if not target_repo:

        errors.append("Repository name is required to create an issue.")

      else:

        st.session_state.built_payloads.append({

          "tool": "create_issue",

          "arguments": {

            "owner": owner,

            "repo": target_repo,

            "title": issue_title,

            "body": issue_body

          }

        })

    # Process pasted LLM JSON into multi_file payload

    if llm_json_text.strip():

      try:

        parsed = parse_json_from_model(llm_json_text)

        if parsed.get("tool") == "multi_file":

          llm_args = parsed.get("arguments", {})

        elif parsed.get("files"):

          llm_args = parsed

        else:

          raise ValueError("Parsed JSON does not look like a multi_file payload.")

        owner_final = llm_args.get("owner", owner)

        repo_final = llm_args.get("repo", target_repo)

        branch_final = llm_args.get("branch", target_branch_for_workflow)

        files = llm_args.get("files", [])

        final_files = []

        for f in files:

          content = f.get("content")

          if content is None and "local_path" in f:

            final_files.append({"path": f["path"], "local_path": f["local_path"]})

          else:

            if content and not is_base64(content) and (send_as_base64 or st.session_state.get("send_base64_default")):

              content = b64_of_text(content)

            final_files.append({"path": f["path"], "content": content})

        st.session_state.built_payloads.append({

          "tool": "multi_file",

          "arguments": {

            "owner": owner_final,

            "repo": repo_final,

            "branch": branch_final,

            "commit_message": llm_args.get("commit_message", "Add generated files"),

            "files": final_files

          }

        })

        st.success("Processed LLM multi_file JSON and added to built payloads.")

      except Exception as e:

        errors.append(f"Failed to parse/process LLM JSON: {e}")

    if errors:

      for e in errors:

        st.error(e)

    else:

      if st.session_state.built_payloads:

        st.success(f"Built {len(st.session_state.built_payloads)} payload(s). Preview below.")

        for i, p in enumerate(st.session_state.built_payloads, start=1):

          st.code(f"Payload #{i}:\n" + json.dumps(p, indent=2))

          try:

            args = p.get("arguments", {})

            path = args.get("path", "")

            content = args.get("content", "")

            if path and (path.endswith(".yml") or path.endswith(".yaml")):

              with st.expander(f"Preview YAML for {path}", expanded=False):

                try:

                  if content and is_base64(content):

                    decoded = base64.b64decode(content).decode("utf-8")

                    st.text_area("YAML", value=decoded, height=320)

                  else:

                    st.text_area("YAML", value=content, height=320)

                except Exception as e:

                  st.write("Error decoding content:", e)

          except Exception as e:

            st.write("Error processing payload preview:", e)

      else:

        st.info("No payloads built (check selections).")

# --- Process LLM multi_file sandbox flow (packs workspace into multi_file) ---

if process_llm_btn and llm_json_text.strip():

  try:

    parsed = parse_json_from_model(llm_json_text)

    if parsed.get("tool") != "multi_file":

      st.error("LLM JSON is not a multi_file tool output.")

    else:

      llm_args = parsed.get("arguments", {})

      model_files = llm_args.get("files", [])

      with ephemeral_workspace() as tmpdir:

        write_files_to_dir(model_files, tmpdir)

        st.info("Files written to ephemeral workspace (preview):")

        for f in model_files:

          st.write(f"- {f.get('path')} (base64: {is_base64(f.get('content'))})")

        run_in_docker = llm_args.get("run_in_docker")

        if run_in_docker:

          st.markdown("**Sandbox commands suggested by model**")

          st.code(json.dumps(run_in_docker, indent=2))

          confirm_run = st.checkbox("Run sandbox commands locally (in Docker) — I understand the risks", value=False)

          if confirm_run:

            image = run_in_docker.get("image", "python:3.11-slim")

            commands = run_in_docker.get("commands", [])

            timeout_seconds = int(run_in_docker.get("timeout_seconds", 120))

            allow_network = bool(run_in_docker.get("allow_network", False))

            st.info(f"Running inside {image} (timeout {timeout_seconds}s). Network allowed: {allow_network}")

            ok, logs, exit_code = run_commands_in_docker(tmpdir, image, commands, timeout_seconds, allow_network)

            st.text_area("Sandbox logs (stdout+stderr)", value=logs, height=400)

            if not ok:

              st.error(f"Sandbox run failed (exit {exit_code}). Inspect logs. Will NOT auto-push unless you continue manually.")

              proceed_after_failure = st.button("Package & add files anyway (I accept risk)")

              if not proceed_after_failure:

                st.stop()

            else:

              st.success("Sandbox commands succeeded.")

        final_files = package_dir_to_files(tmpdir)

        owner_final = llm_args.get("owner", owner)

        repo_final = llm_args.get("repo", repo_name or owner_final)

        branch_final = llm_args.get("branch", branch_name or "main")

        multi_payload = {

          "tool": "multi_file",

          "arguments": {

            "owner": owner_final,

            "repo": repo_final,

            "branch": branch_final,

            "commit_message": llm_args.get("commit_message", "Initial commit (generated)"),

            "files": final_files

          }

        }

        st.session_state.built_payloads.append(multi_payload)

        st.success("Packaged sandbox workspace and added as multi_file payload.")

  except Exception as e:

    st.error(f"LLM processing failed: {e}")

# --- Wiki / docs generation for existing repo ---

st.markdown("---")

st.markdown("## Wiki / Docs generation (existing repo)")

st.write("Generate wiki pages for an existing repository by scanning files and asking Azure OpenAI to produce Markdown pages.")

op_generate_wiki_existing = st.checkbox("Generate wiki for an existing repository (scan repo)", value=False)

if op_generate_wiki_existing:

  existing_owner = st.text_input("Existing repo owner (for scanning)", value=owner)

  existing_repo = st.text_input("Existing repo name (for scanning)", value=repo_name or "")

  existing_branch = st.text_input("Branch to scan (existing)", value=branch_name or "main")

  keep_existing_files = st.checkbox("Include README & top files content in model prompt (recommended)", value=True)

  auto_push_wiki = st.checkbox("Auto-publish generated pages to GitHub Wiki (requires git + PAT)", value=True)

  build_wiki_btn = st.button("Build wiki payloads (scan repo & generate docs)")

  send_wiki_only_btn = st.button("Send wiki-only payloads to MCP proxy / publish to wiki")

  if build_wiki_btn:

    st.session_state.wiki_payloads = []

    if not existing_owner or not existing_repo:

      st.error("Please provide existing repo owner and name to scan.")

    else:

      try:

        st.info("Scanning repository tree...")

        tree = fetch_repo_tree(existing_owner, existing_repo, existing_branch)

        entries = tree.get("tree", [])

        important_files = []

        for item in entries:

          path = item.get("path", "")

          if path.startswith(".git") or item.get("type") != "blob":

            continue

          if any(path.lower().endswith(s) for s in ["readme.md", "package.json", "requirements.txt", "pyproject.toml", "setup.py", "main.py", "index.js"]):

            important_files.append(path)

          if len(important_files) >= 30:

            break

        file_summaries = []

        if keep_existing_files:

          for p in important_files:

            snippet = fetch_file_content(existing_owner, existing_repo, p, existing_branch)

            file_summaries.append({"path": p, "snippet": (snippet[:5000] + ("... (truncated)" if len(snippet) > 5000 else ""))})

        else:

          file_summaries = [{"path": p, "snippet": ""} for p in important_files]

        # Ask Azure to create pages

        system_prompt = (

          "You are a documentation writer assistant. Given a repository file listing and a short "

          "collection of important file contents/snippets, produce a set of Markdown docs that "

          "explain the project, how to run it, architecture overview, and a 'Getting Started' page. "

          "Return JSON only with the format: {\"pages\":[{\"path\":\"Home.md\",\"content\":\"...\"}, ...] }"

        )

        user_payload = {

          "owner": existing_owner,

          "repo": existing_repo,

          "branch": existing_branch,

          "file_summaries": file_summaries,

          "instruction": prompt_text or "Create docs for this repository suitable for wiki pages."

        }

        st.info("Requesting docs from Azure OpenAI...")

        model_input = system_prompt + "\n\n" + json.dumps(user_payload, indent=2)

        raw = azure_chat_completion(model_input, temperature=0.0, max_tokens=2000)

        st.text_area("Raw model output (debug)", value=raw, height=200)

        parsed = parse_json_from_model(raw)

        pages = parsed.get("pages", [])

        if not pages:

          st.error("Model returned no pages. Check raw output above.")

        else:

          for page in pages:

            path = page.get("path")

            content = page.get("content", "")

            if not path or content is None:

              continue

            payload = {

              "tool": "create_or_update_file",

              "arguments": {

                "owner": existing_owner,

                "repo": existing_repo,

                "branch": existing_branch,

                "path": path,

                "message": f"Add docs: {path}",

                # For create_or_update_file we will keep content plain unless user asked base64

                "content": content if not (send_as_base64 or st.session_state.get("send_base64_default")) else b64_of_text(content)

              }

            }

            st.session_state.wiki_payloads.append(payload)

          st.success(f"Built {len(st.session_state.wiki_payloads)} wiki payload(s). Preview below.")

          for i, p in enumerate(st.session_state.wiki_payloads, start=1):

            st.code(f"Wiki Payload #{i}:\n" + json.dumps(p, indent=2))

      except Exception as e:

        st.error(f"Failed to build wiki payloads: {e}")

  if send_wiki_only_btn:

    if not st.session_state.get("wiki_payloads"):

      st.error("No wiki payloads available. Click 'Build wiki payloads' first.")

    else:

      # Option: auto-publish via wiki.git if requested and git present

      if auto_push_wiki:

        if not shutil.which("git"):

          st.error("git is not available on this machine; cannot auto-publish wiki. Please install git.")

        elif not GITHUB_PAT:

          st.error("GITHUB_PERSONAL_ACCESS_TOKEN is not set in environment; cannot push to wiki.")

        else:

          # Convert payloads to plain-text files

          files_to_push = []

          for p in st.session_state.wiki_payloads:

            args = p.get("arguments", {})

            path = args.get("path")

            content_b64_or_text = args.get("content", "")

            if is_base64(content_b64_or_text):

              try:

                content_plain = base64.b64decode(content_b64_or_text).decode("utf-8")

              except Exception:

                content_plain = content_b64_or_text

            else:

              content_plain = content_b64_or_text

            files_to_push.append({"path": path, "content": content_plain})

          st.info("Publishing pages to GitHub Wiki (this clones and pushes to <repo>.wiki.git)...")

          ok, logs = push_files_to_wiki_git(existing_owner, existing_repo, files_to_push, commit_message="Add generated wiki pages (MCP UI)")

          st.text_area("Git push logs", value=logs, height=400)

          if ok:

            st.success("Wiki published successfully. Check the repository's Wiki tab.")

          else:

            st.error("Wiki publish failed. See logs above.")

      else:

        # Just send create_or_update_file payloads to MCP (these will add files into repo, not wiki)

        wiki_results = []

        for idx, payload in enumerate(st.session_state.wiki_payloads, start=1):

          st.write(f"Sending wiki payload #{idx} -> tool: {payload.get('tool')}")

          try:

            resp = requests.post(mcp_url, json=payload, timeout=180)

          except Exception as e:

            wiki_results.append({"error": f"Request failed: {e}"})

            st.error(f"Request failed for wiki payload #{idx}: {e}")

            continue

          try:

            jr = resp.json()

            wiki_results.append({"status": resp.status_code, "json": jr})

          except Exception:

            wiki_results.append({"status": resp.status_code, "text": resp.text})

        st.subheader("Wiki send results")

        st.write(wiki_results)

        succ = [r for r in wiki_results if r.get("status") and int(r.get("status")) < 400]

        fail = [r for r in wiki_results if (r.get("status") and int(r.get("status")) >= 400) or r.get("error")]
        st.success(f"Wiki: {len(succ)} succeeded, {len(fail)} failed")
st.markdown("---")

# --- Send payloads logic (main) ---

if send_pressed:
  if not st.session_state.get("built_payloads"):
    st.error("No built payloads found. Click 'Build payloads' first.")
  else:
    results = []
    for idx, payload in enumerate(st.session_state.built_payloads, start=1):
      st.write(f"Sending payload #{idx} -> tool: {payload.get('tool')}")
      # If multi_file, we may need to ensure files have proper content encoding based on UI choice:
      p = payload.copy()
      if p.get("tool") == "multi_file":
        args = p.get("arguments", {})
        files = args.get("files", [])
        new_files = []
        for f in files:
          content = f.get("content")
          if content is None and "local_path" in f:
            new_files.append(f) # server code may support local_path -> but likely not over http proxy
          else:
            if send_as_base64 or st.session_state.get("send_base64_default"):
              if not is_base64(content):
                content = b64_of_text(content)
            else:
              if is_base64(content):
                # we decode base64 to plain if UI requested plain sending
                try:
                  content = base64.b64decode(content).decode("utf-8")
                except Exception:
                  pass
            new_files.append({"path": f.get("path"), "content": content})
        p["arguments"]["files"] = new_files

      # For create_or_update_file ensure content respects the send_as_base64 toggle:

      if p.get("tool") == "create_or_update_file":
        args = p.get("arguments", {})
        content = args.get("content")
        if send_as_base64 or st.session_state.get("send_base64_default"):
          if content and not is_base64(content):
            args["content"] = b64_of_text(content)
        else:
          if content and is_base64(content):
            try:
              args["content"] = base64.b64decode(content).decode("utf-8")
            except Exception:
              pass
        p["arguments"] = args
      try:
        resp = requests.post(mcp_url, json=p, timeout=120)
      except Exception as e:
        results.append({"error": f"Request failed: {e}"})
        st.error(f"Request failed for payload #{idx}: {e}")
        continue
      try:
        jr = resp.json()
        results.append({"status": resp.status_code, "json": jr})
      except Exception:
        results.append({"status": resp.status_code, "text": resp.text})

    st.subheader("Results")
    st.write(results)
    successes = [r for r in results if r.get("status") and int(r.get("status")) < 400]
    failures = [r for r in results if (r.get("status") and int(r.get("status")) >= 400) or r.get("error")]
    st.success(f"{len(successes)} succeeded, {len(failures)} failed")

st.markdown("---")
st.markdown("Notes:")
st.markdown("- The app builds the JSON the http-proxy expects. For safe compatibility some MCP builds require file content base64; toggle the 'Send file content as base64' checkbox if your MCP expects base64. Default here is plain text (not base64) as requested.")
st.markdown("- To auto-publish wiki pages the machine running Streamlit must have `git` installed and `GITHUB_PERSONAL_ACCESS_TOKEN` set and valid for pushing to the repo's wiki.")
st.markdown("- If you get Azure 404 errors, verify your Azure deployment name, endpoint and API-version.")
