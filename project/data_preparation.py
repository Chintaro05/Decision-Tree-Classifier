"""
Data Preparation: Download and clean dataset from The Stack (bigcode/the-stack-dedup)

FIX 1: Corrected API call — uses data_dir=f"data/{stack_dir}" instead of name=lang
FIX 2: Added LANG_TO_STACK_DIR mapping with verified directory names
FIX 3: Loads all languages (removed the [:5] limitation from previous version)
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from collections import defaultdict
import random
from feature_engineering import FeatureExtractor

# Set random seed for reproducibility
random.seed(42)

# -----------------------------------------------------------------------
# Mapping: internal language name -> actual directory name in The Stack
# Verified from bigcode/the-stack-dedup dataset card
# -----------------------------------------------------------------------
LANG_TO_STACK_DIR = {
    "python":     "python",
    "java":       "java",
    "javascript": "javascript",
    "typescript": "typescript",
    "csharp":     "c-sharp",
    "cpp":        "cpp",
    "c":          "c",
    "ruby":       "ruby",
    "php":        "php",
    "go":         "go",
    "rust":       "rust",
    "kotlin":     "kotlin",
    "swift":      "swift",
    "scala":      "scala",
    "haskell":    "haskell",
    "r":          "r",
    "perl":       "perl",
    "lua":        "lua",
    "shell":      "shell",
    "sql":        "sql",
    "html":       "html",
    "css":        "css",
    "xml":        "xml",
    "markdown":   "markdown",
    "tex":        "tex",
    "dockerfile": "dockerfile",
    "makefile":   "makefile",
    "julia":      "julia",
    "bash":       "shell",        # bash scripts stored under shell
    "powershell": "powershell",
    "dart":       "dart",
    "groovy":     "groovy",
    "jsx":        "jsx",
    "tsx":        "typescript",   # tsx stored under typescript
    # --- UNKNOWN formats (held-out, NOT trained on) ---
    "json":       "json",
    "yaml":       "yaml",
    "csv":        "csv",
    "toml":       "toml",
    "ini":        "ini",
    "svg":        "svg",
}

# 34 KNOWN languages used for training
KNOWN_LANGUAGES = [
    "python", "java", "javascript", "typescript", "csharp",
    "cpp", "c", "ruby", "php", "go",
    "rust", "kotlin", "swift", "scala", "haskell",
    "r", "perl", "lua", "shell", "sql",
    "html", "css", "xml", "markdown", "tex",
    "dockerfile", "makefile", "julia", "bash", "powershell",
    "dart", "groovy", "jsx", "tsx"
]

# 6 UNKNOWN formats — held out, used ONLY to test unknown detection
UNKNOWN_LANGUAGES = ["json", "yaml", "csv", "toml", "ini", "svg"]

LANGUAGES = KNOWN_LANGUAGES + UNKNOWN_LANGUAGES


def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    return text.strip()


def is_clean_sample(content, seen):
    """Validate a code sample — filter binary, duplicates, and noisy text."""
    if not isinstance(content, str):
        return False

    text = content.strip()
    if len(text) < 50:
        return False
    if len(text) > 20000:
        return False

    lines = text.split('\n')
    if len(lines) > 500:
        return False

    # Drop samples with overly long individual lines (minified/base64)
    if max((len(line) for line in lines[:200]), default=0) > 500:
        return False

    # Drop binary-like content with too many control characters
    sample = text[:1000]
    nonprint = sum(1 for ch in sample if ord(ch) < 9 or 13 < ord(ch) < 32)
    if nonprint > 50:
        return False

    # Drop duplicates
    key = hash(text[:500])
    if key in seen:
        return False
    seen.add(key)

    return True


def truncate_file(content, max_lines=100):
    """Keep first N lines of file — headers are the most discriminative."""
    if not isinstance(content, str):
        return ""
    lines = content.split('\n')
    return '\n'.join(lines[:max_lines])


def is_mislabeled_json(content, language):
    """Drop JS/TS samples that look more like JSON than real code."""
    if language not in {"javascript", "typescript", "jsx", "tsx"}:
        return False
    return FeatureExtractor.json_like_score(content) > 0.8


def _load_from_stack(lang, stack_dir, samples_per_class, seen):
    """Stream samples for one language from the-stack-dedup."""
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{stack_dir}",   # FIX: correct API parameter
        split="train",
        streaming=True
    )

    results = []
    for sample in ds:
        if len(results) >= samples_per_class:
            break

        content = sample.get('content', '')
        content = clean_text(content)

        if not is_clean_sample(content, seen):
            continue
        content = truncate_file(content)

        # Filter mislabeled JSON-like content in JS/TS files
        if is_mislabeled_json(content, lang):
            continue

        results.append({
            'language': lang,
            'content': content,
            'file_id': f"{lang}_{len(results)}"
        })

    return results


def download_dataset(languages=LANGUAGES, samples_per_class=1000, output_dir="data"):
    """
    Download dataset from bigcode/the-stack-dedup.
    Falls back to synthetic templates if download fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading dataset — {len(languages)} languages, "
          f"{samples_per_class} samples each...")

    data_dict = defaultdict(list)
    seen = set()
    failed = []

    for lang in languages:
        stack_dir = LANG_TO_STACK_DIR.get(lang, lang)
        print(f"  Loading {lang} (data/{stack_dir}) ...")
        try:
            samples = _load_from_stack(lang, stack_dir, samples_per_class, seen)
            data_dict[lang] = samples
            print(f"    ✓ {len(samples)} valid samples")
        except Exception as e:
            print(f"    ✗ Failed ({e}) — will use synthetic fallback")
            failed.append(lang)

    # Fallback: fill missing languages with synthetic data
    if failed:
        print(f"\nUsing synthetic fallback for: {failed}")
        synthetic = create_synthetic_dataset(failed, samples_per_class)
        for lang, samples in synthetic.items():
            data_dict[lang].extend(samples)

    # Final check: if still insufficient, full synthetic run
    total = sum(len(v) for v in data_dict.values())
    if total < len(languages) * 20:
        print("Insufficient real data — using full synthetic dataset.")
        data_dict = create_synthetic_dataset(languages, samples_per_class)

    all_data = []
    for lang_samples in data_dict.values():
        all_data.extend(lang_samples)

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['content'], keep='first')
    df.to_csv(os.path.join(output_dir, "raw_dataset.csv"), index=False)
    print(f"\n✓ Saved {len(df)} samples to raw_dataset.csv")
    print("Distribution:")
    print(df['language'].value_counts().to_string())
    return df


def create_synthetic_dataset(languages, samples_per_class):
    """Create synthetic dataset with realistic code snippets (fallback only)."""

    templates = {
        "python": [
            "def hello():\n    print('Hello, World!')\n    return True",
            "import numpy as np\ndata = np.array([1, 2, 3])\nresult = np.mean(data)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
            "import pandas as pd\ndf = pd.read_csv('file.csv')\ndf.head()",
        ],
        "java": [
            'public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}',
            "public interface Service {\n    void execute();\n}",
            "public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}",
            'ArrayList<String> list = new ArrayList<>();\nlist.add("item");',
            "try {\n    int x = 10 / 0;\n} catch (Exception e) {\n    e.printStackTrace();\n}",
        ],
        "javascript": [
            "function hello() {\n    console.log('Hello, World!');\n    return true;\n}",
            "const arr = [1, 2, 3];\nconst doubled = arr.map(x => x * 2);",
            "async function fetchData() {\n    const response = await fetch('/api/data');\n    return response.json();\n}",
            "class Calculator {\n    add(a, b) {\n        return a + b;\n    }\n}",
            "const obj = { name: 'John', age: 30 };\nconst {name, age} = obj;",
        ],
        "typescript": [
            "function greet(name: string): string {\n    return `Hello, ${name}`;\n}",
            "interface User {\n    id: number;\n    name: string;\n}",
            "const arr: number[] = [1, 2, 3];\nconst doubled = arr.map((x: number) => x * 2);",
        ],
        "csharp": [
            'using System;\nnamespace Demo {\n    class Program {\n        static void Main() {\n            Console.WriteLine("Hello");\n        }\n    }\n}',
            "public class Calculator {\n    public int Add(int a, int b) => a + b;\n}",
        ],
        "cpp": [
            '#include <iostream>\nusing namespace std;\nint main() {\n    cout << "Hello" << endl;\n    return 0;\n}',
            "#include <vector>\nvector<int> v = {1, 2, 3};\nfor(int x : v) cout << x;",
        ],
        "c": [
            '#include <stdio.h>\nint main(void) {\n    printf("Hello\\n");\n    return 0;\n}',
            "#include <stdlib.h>\nint* arr = malloc(10 * sizeof(int));\nfree(arr);",
        ],
        "ruby": [
            "def greet(name)\n  puts \"Hello, #{name}\"\nend\ngreet('World')",
            "class Calculator\n  def add(a, b)\n    a + b\n  end\nend",
        ],
        "php": [
            "<?php\nfunction greet($name) {\n    echo \"Hello, $name\";\n}\ngreet('World');\n?>",
            "<?php\nclass Calculator {\n    public function add($a, $b) {\n        return $a + $b;\n    }\n}\n?>",
        ],
        "go": [
            'package main\nimport "fmt"\nfunc main() {\n    fmt.Println("Hello")\n}',
            "func add(a, b int) int {\n    return a + b\n}",
        ],
        "rust": [
            "fn main() {\n    let x: i32 = 10;\n    println!(\"{}\", x);\n}",
            "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
        ],
        "kotlin": [
            "fun main() {\n    val x = 10\n    println(\"Hello $x\")\n}",
            "data class User(val name: String, val age: Int)",
        ],
        "swift": [
            "func greet(name: String) {\n    print(\"Hello \\(name)\")\n}",
            "struct Calculator {\n    func add(_ a: Int, _ b: Int) -> Int { a + b }\n}",
        ],
        "scala": [
            "object Main {\n  def main(args: Array[String]): Unit = {\n    println(\"Hello\")\n  }\n}",
            "case class User(name: String, age: Int)",
        ],
        "haskell": [
            "module Main where\nmain :: IO ()\nmain = putStrLn \"Hello\"",
            "add :: Int -> Int -> Int\nadd x y = x + y",
        ],
        "r": [
            "greet <- function(name) {\n  cat(\"Hello\", name)\n}\nx <- c(1, 2, 3)",
            "library(ggplot2)\ndf <- data.frame(x=1:5, y=1:5)\nggplot(df, aes(x,y)) + geom_point()",
        ],
        "perl": [
            "#!/usr/bin/perl\nuse strict;\nmy $x = 10;\nprint \"Hello $x\\n\";",
            "sub greet {\n    my $name = shift;\n    print \"Hello, $name\\n\";\n}",
        ],
        "lua": [
            "local function greet(name)\n  print(\"Hello \" .. name)\nend\ngreet(\"World\")",
            "for i = 1, 10 do\n  if i % 2 == 0 then print(i) end\nend",
        ],
        "shell": [
            "#!/bin/bash\nfor file in *.txt; do\n  echo \"Processing $file\"\ndone",
            "#!/bin/sh\nif [ -f \"/tmp/data.csv\" ]; then\n  cat /tmp/data.csv\nfi",
        ],
        "sql": [
            "SELECT id, name FROM users\nWHERE age > 18\n-- filter by age\nORDER BY name;",
            "INSERT INTO users (name, email)\nVALUES ('John', 'john@example.com');",
        ],
        "html": [
            "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n<h1>Welcome</h1>\n</body>\n</html>",
            "<div class=\"container\">\n    <p>Content here</p>\n    <ul><li>Item 1</li></ul>\n</div>",
        ],
        "css": [
            ".container {\n    display: flex;\n    justify-content: center;\n}\n.item {\n    color: blue;\n}",
            "body {\n    margin: 0;\n    padding: 0;\n    font-family: Arial;\n}\na:hover {\n    text-decoration: underline;\n}",
        ],
        "xml": [
            '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n    <item id="1">Value</item>\n</root>',
            "<config>\n    <server host=\"localhost\" port=\"8080\"/>\n    <database name=\"mydb\"/>\n</config>",
        ],
        "markdown": [
            "# Title\n## Subtitle\nThis is **bold** and *italic* text.",
            "- Item 1\n- Item 2\n  - Nested item\n\n```python\ncode block\n```",
        ],
        "tex": [
            "\\documentclass{article}\n\\begin{document}\n% comment\nHello $x^2$.\n\\end{document}",
            "\\section{Introduction}\nThis paper discusses $\\alpha$ and $\\beta$.\n\\begin{equation}\nE = mc^2\n\\end{equation}",
        ],
        "dockerfile": [
            "FROM python:3.11\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]",
            "FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y python3\nEXPOSE 8080",
        ],
        "makefile": [
            ".PHONY: build run\nbuild:\n\tgcc -o app main.c\nrun:\n\t./app\nclean:\n\trm -f app",
        ],
        "julia": [
            "function greet(name)\n    println(\"Hello $name\")\nend\nx = [1, 2, 3]",
            "struct Point\n    x::Float64\n    y::Float64\nend",
        ],
        "bash": [
            "#!/bin/bash\nset -e\nfor i in 1 2 3; do\n  echo \"Step $i\"\ndone",
            "#!/bin/bash\nif [[ $# -eq 0 ]]; then\n  echo \"Usage: $0 <name>\"\n  exit 1\nfi",
        ],
        "powershell": [
            "$name = \"World\"\nfunction Greet { param($n) Write-Host \"Hello $n\" }\nGreet $name",
            "Get-ChildItem -Path . | Where-Object { $_.Extension -eq '.txt' }",
        ],
        "dart": [
            "void main() {\n  var x = 10;\n  print('Hello $x');\n}",
            "class Person {\n  String name;\n  Person(this.name);\n}",
        ],
        "groovy": [
            "def greet(name) {\n  println \"Hello $name\"\n}\ngreet('World')",
            "def list = [1, 2, 3]\nlist.each { println it }",
        ],
        "jsx": [
            "import React from 'react';\nconst App = () => <div>Hello JSX</div>;\nexport default App;",
            "const Button = ({ onClick, label }) => (\n  <button onClick={onClick}>{label}</button>\n);",
        ],
        "tsx": [
            "import React from 'react';\ninterface Props { name: string; }\nconst App: React.FC<Props> = ({ name }) => <div>{name}</div>;\nexport default App;",
            "const value: number = 42;\nconst message: string = `Value is ${value}`;",
        ],
        # UNKNOWN formats
        "json": [
            '{\n  "name": "John",\n  "age": 30,\n  "items": [1, 2, 3]\n}',
            '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',
        ],
        "yaml": [
            "name: demo\nversion: 1\nitems:\n  - one\n  - two\nnested:\n  key: value",
            "services:\n  web:\n    image: nginx\n    ports:\n      - \"80:80\"",
        ],
        "csv": [
            "id,name,age,city\n1,Alice,30,NYC\n2,Bob,25,LA\n3,Carol,35,SF",
            "date,open,high,low,close\n2024-01-01,100,110,95,105",
        ],
        "toml": [
            '[package]\nname = "demo"\nversion = "1.0"\n\n[dependencies]\nfoo = "^2.0"',
            "[server]\nhost = \"localhost\"\nport = 8080\ndebug = true",
        ],
        "ini": [
            "[section]\nkey = value\n; comment\nother = 123\n[next]\nflag = true",
            "[database]\nhost=localhost\nport=5432\nname=mydb",
        ],
        "svg": [
            '<svg xmlns="http://www.w3.org/2000/svg" width="100">\n  <circle cx="50" cy="50" r="40" fill="red"/>\n</svg>',
            '<svg width="200" height="200">\n  <rect x="10" y="10" width="80" height="80" fill="blue"/>\n</svg>',
        ],
    }

    def random_comment(lang):
        comments = {
            "python": ["# compute result", "# parse input", "# TODO: add tests"],
            "javascript": ["// update state", "// fetch data"],
            "java": ["// initialize service", "// TODO implement"],
            "bash": ["# run pipeline", "# backup files"],
            "powershell": ["# list files", "# check status"],
            "sql": ["-- create table", "-- select rows"],
            "html": ["<!-- main content -->", "<!-- header -->"],
            "css": ["/* style wrapper */", "/* responsive layout */"],
        }
        return random.choice(comments.get(lang, ["# sample code"]))

    data_dict = defaultdict(list)
    for lang in languages:
        template_list = templates.get(lang, [f"# {lang} code\nprint('{lang}')"])
        for i in range(samples_per_class):
            template = random.choice(template_list)
            content = template
            if random.random() < 0.5:
                content += "\n" + random_comment(lang)
            content = content.strip()
            data_dict[lang].append({
                'language': lang,
                'content': content,
                'file_id': f"{lang}_{i}"
            })
    return data_dict


if __name__ == "__main__":
    df = download_dataset()
    print("\nDataset Statistics:")
    print(df['language'].value_counts())
