# `argparse` Module in Python

The `argparse` module is a standard library in Python used to create **command-line interfaces (CLI)**. It parses command-line arguments, generates help messages, validates inputs, and makes it easy to build user-friendly command-line tools.

---

## Key Concepts

### ArgumentParser

* Core class for handling arguments.
* Provides methods to add, parse, and manage arguments.

### Arguments

* **Positional arguments**: Required arguments identified by their order.
* **Optional arguments**: Prefixed with `-` or `--`, can have default values.
* **Flags**: Boolean arguments that toggle a feature (e.g., `--verbose`).

### Actions

* Define how command-line arguments should be handled:

  * `store`: Default, stores the value.
  * `store_const`: Stores a constant value.
  * `store_true` / `store_false`: Stores `True`/`False`.
  * `append`: Appends multiple values into a list.
  * `count`: Counts how many times a flag appears.

### Argument Types

* Automatically converts input to a specific type (`int`, `float`, `str`, custom functions).

### Default Values

* Arguments can have defaults when not provided.

### Help & Usage

* Automatically generates `--help` text.
* Customizable usage messages.

### Subcommands

* Support for nested commands (like `git add`, `git commit`).

---

## Common Methods

* `ArgumentParser(description="...")` → Create parser.
* `add_argument(name/flag, **kwargs)` → Add arguments.

  * `type` → Type conversion.
  * `default` → Default value.
  * `required` → Mark argument as required.
  * `help` → Add help text.
  * `choices` → Restrict to given values.
* `parse_args()` → Parse CLI arguments into a Namespace.
* `print_help()` → Show help message.
* `print_usage()` → Show usage message.

---

## Examples

### Basic Positional and Optional Arguments

```python
import argparse

parser = argparse.ArgumentParser(description="Demo CLI tool")
parser.add_argument("filename", help="Input file name")  
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")  
parser.add_argument("-n", "--number", type=int, default=1, help="Number of iterations")  

args = parser.parse_args()

print(f"File: {args.filename}")
print(f"Verbose: {args.verbose}")
print(f"Number: {args.number}")
```

**Run in CLI**:

```bash
python script.py data.txt -v -n 5
```

---

### Using `choices` and `default`

```python
parser.add_argument(
    "--mode",
    choices=["fast", "slow", "medium"],
    default="medium",
    help="Execution mode"
)
```

---

### Subcommands Example

```python
subparsers = parser.add_subparsers(dest="command")

# Subcommand 'add'
add_parser = subparsers.add_parser("add", help="Add a new item")
add_parser.add_argument("item", help="Item to add")

# Subcommand 'remove'
remove_parser = subparsers.add_parser("remove", help="Remove an item")
remove_parser.add_argument("item", help="Item to remove")

args = parser.parse_args()
if args.command == "add":
    print(f"Adding {args.item}")
elif args.command == "remove":
    print(f"Removing {args.item}")
```

---

## Usage Scenarios

* Writing CLI tools (e.g., `pip`, `django-admin`).
* Automating scripts with configurable parameters.
* Handling complex subcommands in one script.

---
