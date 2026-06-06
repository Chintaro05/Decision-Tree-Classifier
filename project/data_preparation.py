"""
Data Preparation: Download and clean dataset from The Stack v2
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

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    return text.strip()


def is_clean_sample(content, seen):
    """Validate a code sample and filter binary-like, duplicates, or extremely noisy text."""
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

    # Drop samples with overly long individual lines.
    if max((len(line) for line in lines[:200]), default=0) > 500:
        return False

    # Drop binary-like content containing too many control characters.
    sample = text[:1000]
    nonprint = sum(1 for ch in sample if ord(ch) < 9 or 13 < ord(ch) < 32)
    if nonprint > 50:
        return False

    # Drop duplicates across the dataset.
    key = hash(text[:500])
    if key in seen:
        return False
    seen.add(key)

    return True


def truncate_file(content, max_lines=100):
    """Keep first N lines of file"""
    if not isinstance(content, str):
        return ""
    lines = content.split('\n')
    return '\n'.join(lines[:max_lines])


def is_mislabeled_json(content, language):
    """Drop JavaScript/TypeScript samples that look more like JSON than real code."""
    if language not in {"javascript", "typescript", "jsx", "tsx"}:
        return False
    return FeatureExtractor.json_like_score(content) > 0.8

def download_dataset(languages=LANGUAGES, samples_per_class=1000, output_dir="data"):
    """
    Download dataset from The Stack v2
    If the remote dataset is unavailable, fallback to synthetic templates.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Attempting to download dataset for {len(languages)} languages...")
    print(f"Target: {samples_per_class} samples per language")
    
    data_dict = defaultdict(list)
    seen = set()
    
    try:
        # Try to load from The Stack v2
        for lang in languages:
            print(f"Loading {lang}...")
            try:
                ds = load_dataset(
                    "bigcode/the-stack", 
                    name=lang,          
                    split="train",
                    streaming=True
                )
                
                count = 0
                for sample in ds:
                    if count >= samples_per_class:
                        break
                    
                    content = sample.get('content', '')
                    
                    # Clean and validate
                    content = clean_text(content)
                    if not is_clean_sample(content, seen):
                        continue
                    
                    # Truncate to first 100 lines
                    content = truncate_file(content)
                    
                    # Drop mislabeled JSON-like content from JS/TS sources
                    if is_mislabeled_json(content, lang):
                        continue
                    
                    data_dict[lang].append({
                        'language': lang,
                        'content': content,
                        'file_id': f"{lang}_{count}"
                    })
                    count += 1
                
                print(f"  ✓ Got {len(data_dict[lang])} valid samples for {lang}")
            except Exception as e:
                print(f"  ✗ Failed to load {lang}: {e}")
                # Continue with other languages
    except Exception as e:
        print(f"Connection issue: {e}")
        print("Creating synthetic dataset from templates...\n")
        data_dict = create_synthetic_dataset(languages, samples_per_class)
    
    # If we got very few samples, use synthetic data
    total_samples = sum(len(v) for v in data_dict.values())
    if total_samples < len(languages) * 50:
        print("Insufficient real data. Using synthetic templates...\n")
        data_dict = create_synthetic_dataset(languages, samples_per_class)
    
    # Save raw data
    all_data = []
    for lang, samples in data_dict.items():
        all_data.extend(samples)
    
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['content'], keep='first')
    df.to_csv(os.path.join(output_dir, "raw_dataset.csv"), index=False)
    print(f"\n✓ Saved {len(df)} samples to raw_dataset.csv")
    
    return df

def create_synthetic_dataset(languages, samples_per_class):
    """Create synthetic dataset with realistic code snippets"""
    
    templates = {
        "python": [
            "def hello():\n    print('Hello, World!')\n    return True",
            "import numpy as np\ndata = np.array([1, 2, 3])\nresult = np.mean(data)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
            "import pandas as pd\ndf = pd.read_csv('file.csv')\ndf.head()",
        ],
        "java": [
            "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
            "public interface Service {\n    void execute();\n}",
            "public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}",
            "ArrayList<String> list = new ArrayList<>();\nlist.add(\"item\");",
            "try {\n    int x = 10 / 0;\n} catch (Exception e) {\n    e.printStackTrace();\n}",
        ],
        "javascript": [
            "function hello() {\n    console.log('Hello, World!');\n    return true;\n}",
            "const arr = [1, 2, 3];\nconst doubled = arr.map(x => x * 2);",
            "async function fetchData() {\n    const response = await fetch('/api/data');\n    return response.json();\n}",
            "class Calculator {\n    add(a, b) {\n        return a + b;\n    }\n}",
            "const obj = { name: 'John', age: 30 };\nconst {name, age} = obj;",
        ],
        "html": [
            "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n<h1>Welcome</h1>\n</body>\n</html>",
            "<div class=\"container\">\n    <p>Content here</p>\n</div>",
            "<?xml version=\"1.0\"?>\n<root>\n    <item>Value</item>\n</root>",
        ],
        "css": [
            ".container {\n    display: flex;\n    justify-content: center;\n}\n.item {\n    color: blue;\n}",
            "body {\n    margin: 0;\n    padding: 0;\n    font-family: Arial;\n}\na:hover {\n    text-decoration: underline;\n}",
        ],
        "json": [
            '{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}',
            '{\"users\": [{\"id\": 1, \"name\": \"Alice\"}, {\"id\": 2, \"name\": \"Bob\"}]}',
        ],
        "yaml": [
            "name: John\nage: 30\ncity: New York",
            "items:\n  - id: 1\n    name: Item 1\n  - id: 2\n    name: Item 2",
        ],
        "sql": [
            "SELECT * FROM users WHERE age > 18;",
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com');",
            "UPDATE products SET price = 99.99 WHERE id = 1;",
        ],
        "markdown": [
            "# Title\n## Subtitle\nThis is **bold** and *italic* text.",
            "- Item 1\n- Item 2\n  - Nested item\n\n```python\ncode block\n```",
        ],
        "dockerfile": [
            "FROM python:3.9\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]",
            "FROM ubuntu:20.04\nRUN apt-get update && apt-get install -y python3",
        ],
        "makefile": [
            ".PHONY: build run\nbuild:\n\tpython setup.py build\nrun:\n\tpython main.py",
        ],
        "bash": [
            "#!/bin/bash\nfor file in *.txt; do\n  echo \"Processing $file\"\ndone",
            "if [ -f \"/tmp/data.csv\" ]; then\n  cat /tmp/data.csv\nfi",
        ],
        "powershell": [
            "Write-Host \"Hello World\"\nGet-ChildItem -Path . | Where-Object { $_.Extension -eq '.txt' }",
            "$data = Import-Csv 'file.csv'\n$data | ForEach-Object { $_.Name }",
        ],
        "dart": [
            "void main() {\n  print('Hello, Dart');\n}",
            "class Person {\n  String name;\n  Person(this.name);\n}",
        ],
        "groovy": [
            "def hello() {\n  println 'Hello from Groovy'\n}",
            "def list = [1, 2, 3]\nlist.each { println it }",
        ],
        "jsx": [
            "import React from 'react';\nconst App = () => <div>Hello JSX</div>;\nexport default App;",
            "const element = <button onClick={() => alert('click')}>Click</button>;",
        ],
        "tsx": [
            "import React from 'react';\ninterface Props { name: string; }\nconst App: React.FC<Props> = ({ name }) => <div>{name}</div>;\nexport default App;",
            "const value: number = 42;\nconst message: string = `Value is ${value}`;",
        ],
        "toml": [
            "[package]\nname = \"example\"\nversion = \"0.1.0\"",
            "[database]\nserver = \"192.168.1.1\nports = [ 8001, 8001, 8002 ]",
        ],
        "ini": [
            "[settings]\nname = example\nenabled = true",
            "[user]\nusername = admin\npassword = secret",
        ],
        "csv": [
            "name,age,city\nJohn,30,New York\nAlice,25,London",
            "id,value\n1,10\n2,20\n3,30",
        ],
        "svg": [
            "<svg width=\"100\" height=\"100\"><circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"red\" /></svg>",
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100\" height=\"100\" fill=\"blue\" /></svg>",
        ],
    }
    
    def random_comment(lang):
        comments = {
            "python": ["# compute result", "# parse input", "# TODO: add tests"],
            "javascript": ["// update state", "// fetch data", "// eslint-disable-next-line"],
            "java": ["// initialize service", "// check null", "// TODO implement"],
            "bash": ["# run pipeline", "# backup files", "# use set -e"],
            "powershell": ["# list files", "# check status", "# log output"],
            "sql": ["-- create table", "-- select rows", "-- drop temp table"],
            "html": ["<!-- main content -->", "<!-- header -->", "<!-- footer -->"],
            "css": ["/* style wrapper */", "/* responsive layout */"],
        }
        return random.choice(comments.get(lang, ["# sample code"]))

    def random_data_line(lang):
        if lang in {"python", "javascript", "java", "typescript", "tsx", "jsx"}:
            return random.choice([
                "const idx = 42;",
                "let count = items.length;",
                "const result = values.reduce((acc, x) => acc + x, 0);",
                "var message = 'ok';",
                "print('done')" if lang == "python" else "console.log('done');",
            ])
        return ""

    data_dict = defaultdict(list)

    for lang in languages:
        template_list = templates.get(lang, [f"# {lang} code example\nprint('{lang}')"])

        for i in range(samples_per_class):
            template = random.choice(template_list)
            content = template
            if random.random() < 0.6:
                content += "\n" + random_comment(lang)
            if random.random() < 0.4:
                content += "\n" + random_data_line(lang)
            if random.random() < 0.3:
                content += "\n" + random_comment(lang)
            extra_blank_lines = random.randint(0, 3)
            content += "\n" * extra_blank_lines
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
