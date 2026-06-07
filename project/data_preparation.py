"""
Data Preparation: Download and clean dataset from The Stack-smol.

Fixes compared with the previous version:
- Uses data_dir=... instead of name=... when calling load_dataset().
- Maps aliases such as cpp -> data/c++, csharp -> data/c-sharp, tex -> data/tex.
- Falls back per-language to synthetic samples when a language/format is absent from The Stack-smol.
- Does not crash if feature_engineering.FeatureExtractor is unavailable.
"""

import os
import json
import hashlib
import random
from collections import defaultdict

import pandas as pd

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from feature_engineering import FeatureExtractor
except Exception:
    FeatureExtractor = None


# Set random seed for reproducibility
random.seed(42)

DATASET_NAME = "bigcode/the-stack-smol"

# Define training languages and held-out unknown formats
KNOWN_LANGUAGES = [
    "python", "java", "javascript", "typescript", "csharp",
    "cpp", "c", "ruby", "php", "go",
    "rust", "kotlin", "swift", "scala", "haskell",
    "r", "perl", "lua", "shell", "sql",
    "html", "css", "xml", "markdown", "tex",
    "dockerfile", "makefile", "julia", "bash", "powershell",
    "dart", "groovy", "jsx", "tsx"
]

UNKNOWN_LANGUAGES = [
    "json", "yaml", "csv", "toml", "ini", "svg"
]

LANGUAGES = KNOWN_LANGUAGES + UNKNOWN_LANGUAGES

# The Stack-smol uses folder names, not builder config names.
# IMPORTANT: pass these through data_dir=..., not name=...
STACK_SMOL_DIR_MAP = {
    "python": "data/python",
    "java": "data/java",
    "javascript": "data/javascript",
    "typescript": "data/typescript",
    "c": "data/c",
    "cpp": "data/c++",
    "go": "data/go",
    "rust": "data/rust",
    "ruby": "data/ruby",
    "php": "data/php",
    "scala": "data/scala",
    "haskell": "data/haskell",
    "lua": "data/lua",
    "perl": "data/perl",
    "shell": "data/shell",
    "bash": "data/shell",          # alias: use shell data for bash-like samples
    "html": "data/html",
    "css": "data/css",
    "sql": "data/sql",
    "markdown": "data/markdown",
    "dockerfile": "data/dockerfile",
    "makefile": "data/makefile",
    "tex": "data/tex",
    "julia": "data/julia",
    "csharp": "data/c-sharp",
    "powershell": "data/powershell",
}

# Formats/languages below are not reliably present in The Stack-smol.
# They are generated locally so the script still creates a complete dataset.
SYNTHETIC_ONLY_LANGUAGES = set(LANGUAGES) - set(STACK_SMOL_DIR_MAP)


def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    return text.strip()


def is_binary_like(text):
    """Detect binary-like or very noisy content."""
    if not isinstance(text, str) or not text:
        return True

    sample = text[:2000]
    nonprint = sum(1 for ch in sample if ord(ch) < 9 or 13 < ord(ch) < 32)
    if nonprint > 50:
        return True

    # Very long continuous base64-like chunks are often bundled/binary data.
    longest_alnum = 0
    current = 0
    for ch in sample:
        if ch.isalnum() or ch in "+/=":
            current += 1
            longest_alnum = max(longest_alnum, current)
        else:
            current = 0
    return longest_alnum > 500


def is_clean_sample(content, seen):
    """Validate a code sample and filter binary-like, duplicates, or extremely noisy text."""
    if not isinstance(content, str):
        return False

    text = content.strip()
    if len(text) < 50:
        return False
    if len(text) > 20000:
        return False

    lines = text.split("\n")
    if len(lines) > 500:
        return False

    # Drop samples with overly long individual lines.
    if max((len(line) for line in lines[:200]), default=0) > 500:
        return False

    if is_binary_like(text):
        return False

    # Drop duplicates across the dataset. Do not use Python hash(), because it is randomized per run.
    key = hashlib.md5(text[:1000].encode("utf-8", errors="ignore")).hexdigest()
    if key in seen:
        return False
    seen.add(key)

    return True


def truncate_file(content, max_lines=100):
    """Keep first N lines of file."""
    if not isinstance(content, str):
        return ""
    lines = content.split("\n")
    return "\n".join(lines[:max_lines])


def fallback_json_like_score(content):
    """Small fallback when FeatureExtractor is not available."""
    if not isinstance(content, str):
        return 0.0

    text = content.strip()
    if not text:
        return 0.0

    try:
        json.loads(text)
        return 1.0
    except Exception:
        pass

    starts_json = text.startswith("{") or text.startswith("[")
    has_json_pairs = text.count('"') >= 4 and ":" in text
    semicolon_count = text.count(";")
    function_markers = ["function ", "=>", "const ", "let ", "var ", "import ", "export "]
    code_marker_count = sum(marker in text for marker in function_markers)

    score = 0.0
    if starts_json:
        score += 0.4
    if has_json_pairs:
        score += 0.4
    if semicolon_count == 0:
        score += 0.1
    if code_marker_count == 0:
        score += 0.1
    return min(score, 1.0)


def is_mislabeled_json(content, language):
    """Drop JavaScript/TypeScript samples that look more like JSON than real code."""
    if language not in {"javascript", "typescript", "jsx", "tsx"}:
        return False

    if FeatureExtractor is not None and hasattr(FeatureExtractor, "json_like_score"):
        try:
            return FeatureExtractor.json_like_score(content) > 0.8
        except Exception:
            pass

    return fallback_json_like_score(content) > 0.8


def collect_real_samples(lang, samples_per_class, seen):
    """Collect clean real samples for one language from The Stack-smol."""
    if load_dataset is None:
        print("  ✗ Package 'datasets' is not available. Using synthetic samples.")
        return []

    data_dir = STACK_SMOL_DIR_MAP.get(lang)
    if data_dir is None:
        print(f"  - {lang}: not available in The Stack-smol. Using synthetic samples.")
        return []

    print(f"Loading {lang} from {DATASET_NAME}/{data_dir}...")

    try:
        ds = load_dataset(
            DATASET_NAME,
            data_dir=data_dir,      # Correct: data_dir, not name
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  ✗ Failed to load {lang}: {e}")
        return []

    samples = []
    scanned = 0
    max_scan = max(samples_per_class * 25, 5000)

    try:
        for sample in ds:
            scanned += 1
            if len(samples) >= samples_per_class:
                break
            if scanned > max_scan:
                break

            content = sample.get("content", "")
            content = clean_text(content)
            if not is_clean_sample(content, seen):
                continue

            content = truncate_file(content)
            if is_mislabeled_json(content, lang):
                continue

            file_id = sample.get("hexsha") or sample.get("max_stars_repo_path") or f"{lang}_{len(samples)}"
            samples.append({
                "language": lang,
                "content": content,
                "file_id": str(file_id),
            })
    except Exception as e:
        print(f"  ✗ Streaming interrupted for {lang}: {e}")

    print(f"  ✓ Got {len(samples)} valid real samples for {lang}")
    return samples


def download_dataset(languages=LANGUAGES, samples_per_class=1000, output_dir="data"):
    """
    Download dataset from The Stack-smol.
    If a language is unavailable or remote loading fails, fill that class with synthetic templates.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Attempting to prepare dataset for {len(languages)} languages/formats...")
    print(f"Target: {samples_per_class} samples per class")
    print(f"Remote dataset: {DATASET_NAME}\n")

    data_dict = defaultdict(list)
    seen = set()
    min_real_threshold = min(50, samples_per_class)

    for lang in languages:
        if lang in SYNTHETIC_ONLY_LANGUAGES:
            print(f"Loading {lang}...")
            print(f"  - {lang}: no stable The Stack-smol folder. Generating synthetic samples.")
            data_dict[lang].extend(create_synthetic_dataset([lang], samples_per_class)[lang])
            continue

        samples = collect_real_samples(lang, samples_per_class, seen)

        if len(samples) < min_real_threshold:
            need = samples_per_class - len(samples)
            print(f"  ! Too few real samples for {lang}. Adding {need} synthetic samples.")
            synthetic = create_synthetic_dataset([lang], need, start_index=len(samples))[lang]
            samples.extend(synthetic)
        elif len(samples) < samples_per_class:
            need = samples_per_class - len(samples)
            print(f"  ! Only got {len(samples)} real samples for {lang}. Adding {need} synthetic samples.")
            synthetic = create_synthetic_dataset([lang], need, start_index=len(samples))[lang]
            samples.extend(synthetic)

        data_dict[lang].extend(samples[:samples_per_class])

    # Save raw data
    all_data = []
    for lang, samples in data_dict.items():
        all_data.extend(samples)

    df = pd.DataFrame(all_data)
    if df.empty:
        raise RuntimeError("No samples were prepared. Check your environment and dependencies.")

    df = df.drop_duplicates(subset=["language", "content"], keep="first")
    output_path = os.path.join(output_dir, "raw_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved {len(df)} samples to {output_path}")
    return df


def create_synthetic_dataset(languages, samples_per_class, start_index=0):
    """Create synthetic dataset with realistic code/config snippets."""

    templates = {
        "python": [
            "def hello(name):\n    print(f'Hello, {name}')\n    return True",
            "import numpy as np\ndata = np.array([1, 2, 3])\nresult = np.mean(data)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
            "import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.head())",
        ],
        "java": [
            "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
            "public interface Service {\n    void execute();\n}",
            "public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}",
        ],
        "javascript": [
            "function hello() {\n    console.log('Hello, World!');\n    return true;\n}",
            "const arr = [1, 2, 3];\nconst doubled = arr.map(x => x * 2);",
            "async function fetchData() {\n    const response = await fetch('/api/data');\n    return response.json();\n}",
        ],
        "typescript": [
            "type User = { id: number; name: string };\nconst user: User = { id: 1, name: 'Alice' };",
            "function add(a: number, b: number): number {\n    return a + b;\n}",
        ],
        "jsx": [
            "import React from 'react';\nconst App = () => <div>Hello JSX</div>;\nexport default App;",
            "const element = <button onClick={() => alert('click')}>Click</button>;",
        ],
        "tsx": [
            "import React from 'react';\ninterface Props { name: string; }\nconst App: React.FC<Props> = ({ name }) => <div>{name}</div>;",
            "const value: number = 42;\nconst message: string = `Value is ${value}`;",
        ],
        "c": [
            "#include <stdio.h>\nint main(void) {\n    printf(\"Hello C\\n\");\n    return 0;\n}",
            "int add(int a, int b) {\n    return a + b;\n}",
        ],
        "cpp": [
            "#include <iostream>\nint main() {\n    std::cout << \"Hello C++\" << std::endl;\n    return 0;\n}",
            "class Calculator {\npublic:\n    int add(int a, int b) { return a + b; }\n};",
        ],
        "csharp": [
            "using System;\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello C#\");\n    }\n}",
            "public class Calculator {\n    public int Add(int a, int b) => a + b;\n}",
        ],
        "go": [
            "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello Go\")\n}",
            "func add(a int, b int) int {\n    return a + b\n}",
        ],
        "rust": [
            "fn main() {\n    println!(\"Hello Rust\");\n}",
            "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
        ],
        "kotlin": [
            "fun main() {\n    println(\"Hello Kotlin\")\n}",
            "data class User(val id: Int, val name: String)",
        ],
        "swift": [
            "import Foundation\nprint(\"Hello Swift\")",
            "struct User {\n    let id: Int\n    let name: String\n}",
        ],
        "ruby": [
            "def hello(name)\n  puts \"Hello #{name}\"\nend",
            "class Calculator\n  def add(a, b)\n    a + b\n  end\nend",
        ],
        "php": [
            "<?php\nfunction hello($name) {\n    echo \"Hello $name\";\n}\n?>",
            "<?php\n$items = [1, 2, 3];\nforeach ($items as $item) { echo $item; }\n?>",
        ],
        "scala": [
            "object Main extends App {\n  println(\"Hello Scala\")\n}",
            "case class User(id: Int, name: String)",
        ],
        "haskell": [
            "main :: IO ()\nmain = putStrLn \"Hello Haskell\"",
            "add :: Int -> Int -> Int\nadd a b = a + b",
        ],
        "r": [
            "values <- c(1, 2, 3, 4)\nmean_value <- mean(values)\nprint(mean_value)",
            "df <- data.frame(name=c('Alice','Bob'), age=c(25,30))\nprint(df)",
        ],
        "perl": [
            "use strict;\nuse warnings;\nprint \"Hello Perl\\n\";",
            "my @items = (1, 2, 3);\nforeach my $item (@items) { print $item; }",
        ],
        "lua": [
            "function hello(name)\n  print('Hello ' .. name)\nend",
            "local items = {1, 2, 3}\nfor i, value in ipairs(items) do\n  print(value)\nend",
        ],
        "shell": [
            "#!/bin/sh\nfor file in *.txt; do\n  echo \"Processing $file\"\ndone",
            "if [ -f \"/tmp/data.csv\" ]; then\n  cat /tmp/data.csv\nfi",
        ],
        "bash": [
            "#!/bin/bash\nset -e\nfor file in *.txt; do\n  echo \"Processing $file\"\ndone",
            "if [[ -f \"/tmp/data.csv\" ]]; then\n  cat /tmp/data.csv\nfi",
        ],
        "powershell": [
            "Write-Host \"Hello World\"\nGet-ChildItem -Path . | Where-Object { $_.Extension -eq '.txt' }",
            "$data = Import-Csv 'file.csv'\n$data | ForEach-Object { $_.Name }",
        ],
        "html": [
            "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n<h1>Welcome</h1>\n</body>\n</html>",
            "<div class=\"container\">\n    <p>Content here</p>\n</div>",
        ],
        "css": [
            ".container {\n    display: flex;\n    justify-content: center;\n}\n.item {\n    color: blue;\n}",
            "body {\n    margin: 0;\n    padding: 0;\n    font-family: Arial, sans-serif;\n}",
        ],
        "xml": [
            "<?xml version=\"1.0\"?>\n<root>\n    <item id=\"1\">Value</item>\n</root>",
            "<configuration>\n  <setting name=\"enabled\">true</setting>\n</configuration>",
        ],
        "json": [
            '{\n  "name": "John",\n  "age": 30,\n  "city": "New York"\n}',
            '{\n  "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]\n}',
        ],
        "yaml": [
            "name: John\nage: 30\ncity: New York",
            "items:\n  - id: 1\n    name: Item 1\n  - id: 2\n    name: Item 2",
        ],
        "toml": [
            "[package]\nname = \"example\"\nversion = \"0.1.0\"",
            "[database]\nserver = \"192.168.1.1\"\nports = [8001, 8002]",
        ],
        "ini": [
            "[settings]\nname = example\nenabled = true",
            "[user]\nusername = admin\nrole = maintainer",
        ],
        "csv": [
            "name,age,city\nJohn,30,New York\nAlice,25,London",
            "id,value,status\n1,10,ok\n2,20,failed\n3,30,ok",
        ],
        "svg": [
            "<svg width=\"100\" height=\"100\"><circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"red\" /></svg>",
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100\" height=\"100\" fill=\"blue\" /></svg>",
        ],
        "sql": [
            "SELECT * FROM users WHERE age > 18;",
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com');",
            "UPDATE products SET price = 99.99 WHERE id = 1;",
        ],
        "markdown": [
            "# Title\n## Subtitle\nThis is **bold** and *italic* text.",
            "- Item 1\n- Item 2\n  - Nested item\n\n```python\nprint('code block')\n```",
        ],
        "tex": [
            "\\documentclass{article}\n\\begin{document}\nHello \\LaTeX{}\n\\end{document}",
            "\\section{Introduction}\nThis is a sample equation: $a^2 + b^2 = c^2$.",
        ],
        "dockerfile": [
            "FROM python:3.11\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]",
            "FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y python3",
        ],
        "makefile": [
            ".PHONY: build run\nbuild:\n\tpython setup.py build\nrun:\n\tpython main.py",
            "CC=gcc\nCFLAGS=-O2\nmain: main.o\n\t$(CC) $(CFLAGS) -o main main.o",
        ],
        "julia": [
            "function add(a, b)\n    return a + b\nend\nprintln(add(1, 2))",
            "using Statistics\nvalues = [1, 2, 3]\nprintln(mean(values))",
        ],
        "dart": [
            "void main() {\n  print('Hello, Dart');\n}",
            "class Person {\n  String name;\n  Person(this.name);\n}",
        ],
        "groovy": [
            "def hello() {\n  println 'Hello from Groovy'\n}",
            "def list = [1, 2, 3]\nlist.each { println it }",
        ],
    }

    def random_comment(lang):
        comments = {
            "python": ["# compute result", "# parse input", "# TODO: add tests"],
            "javascript": ["// update state", "// fetch data", "// eslint-disable-next-line"],
            "typescript": ["// typed helper", "// validate input", "// TODO refactor"],
            "java": ["// initialize service", "// check null", "// TODO implement"],
            "c": ["// allocate buffer", "// validate pointer", "// return status"],
            "cpp": ["// initialize vector", "// handle exception", "// compute result"],
            "bash": ["# run pipeline", "# backup files", "# use set -e"],
            "shell": ["# run pipeline", "# backup files", "# POSIX shell"],
            "powershell": ["# list files", "# check status", "# log output"],
            "sql": ["-- create table", "-- select rows", "-- drop temp table"],
            "html": ["<!-- main content -->", "<!-- header -->", "<!-- footer -->"],
            "css": ["/* style wrapper */", "/* responsive layout */"],
            "tex": ["% theorem statement", "% compile with pdflatex"],
        }
        return random.choice(comments.get(lang, [f"# synthetic {lang} sample"]))

    def random_data_line(lang, i):
        if lang == "python":
            return f"value_{i} = {i}"
        if lang in {"javascript", "typescript", "jsx", "tsx"}:
            return f"const idx{i} = {i};"
        if lang == "java":
            return f"int value{i} = {i};"
        if lang in {"c", "cpp", "csharp"}:
            return f"int value_{i} = {i};"
        if lang in {"json"}:
            return ""
        if lang in {"yaml", "toml", "ini", "csv", "svg", "xml", "html", "css", "markdown", "tex"}:
            return ""
        return ""

    data_dict = defaultdict(list)

    for lang in languages:
        template_list = templates.get(lang, [f"# {lang} code example\nprint('{lang}')"])

        for offset in range(samples_per_class):
            i = start_index + offset
            template = random.choice(template_list)
            content = template

            # Add controlled variation so synthetic samples are not exact duplicates.
            if lang in {"json", "yaml", "toml", "ini", "csv"}:
                content += f"\n# sample_id: {i}" if lang != "json" else f"\n"
            elif lang == "svg":
                content += f"\n<!-- sample_id: {i} -->"
            elif lang == "xml":
                content += f"\n<!-- sample_id: {i} -->"
            else:
                if random.random() < 0.7:
                    content += "\n" + random_comment(lang)
                extra_line = random_data_line(lang, i)
                if extra_line:
                    content += "\n" + extra_line
                if random.random() < 0.3:
                    content += "\n" + random_comment(lang)

            content = content.strip()
            data_dict[lang].append({
                "language": lang,
                "content": content,
                "file_id": f"{lang}_{i}",
            })

    return data_dict


def get_unknown_samples(fmt_name, snippet, n):
    """Generate unknown-format snippets for calibration."""
    rng = random.Random(hashlib.md5(fmt_name.encode()).hexdigest())
    out = []
    for i in range(n):
        pad = "\n".join(f"// line {rng.randint(0, 9999)}" for _ in range(rng.randint(0, 3)))
        out.append(snippet + ("\n" + pad if pad else ""))
    return out


UNKNOWN_SNIPPETS = {
    "JSON": '{\n  "name": "demo",\n  "version": 1,\n  "items": [1,2,3]\n}\n',
    "YAML": "name: demo\nversion: 1\nitems:\n  - one\n  - two\n",
    "CSV": "id,name,age\n1,Alice,30\n2,Bob,25\n3,Carol,35\n",
    "TOML": '[package]\nname = "demo"\nversion = "1.0"\n',
    "INI": "[section]\nkey = value\n; comment\nflag = true\n",
    "XML": '<?xml version="1.0"?>\n<root>\n  <item id="1">hello</item>\n</root>\n',
    "SVG": '<svg xmlns="http://www.w3.org/2000/svg" width="100">\n  <circle cx="50" cy="50" r="40"/>\n</svg>\n',
}


if __name__ == "__main__":
    df = download_dataset()
    print("\nDataset Statistics:")
    print(df["language"].value_counts())
