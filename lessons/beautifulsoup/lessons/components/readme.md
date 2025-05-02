## Components of BeautifulSoup (Modules and Submodules)

### Top-Level Module

* `bs4`
  The main module that contains all public classes and functions.

---

### Main Components (Modules / Classes under `bs4`)

| Component               | Type  | Description                                                    |
| ----------------------- | ----- | -------------------------------------------------------------- |
| `BeautifulSoup`         | Class | Main class used to create a parse tree from HTML/XML input.    |
| `Tag`                   | Class | Represents an HTML or XML tag in the parse tree.               |
| `NavigableString`       | Class | Represents a string within a tag.                              |
| `Comment`               | Class | Represents a comment node.                                     |
| `Doctype`               | Class | Represents a document type declaration.                        |
| `CData`                 | Class | Represents a CDATA section (in XML).                           |
| `ProcessingInstruction` | Class | Represents processing instructions (for XML).                  |
| `SoupStrainer`          | Class | Used to filter elements while parsing.                         |
| `ResultSet`             | Class | List-like object returned by `find_all()` and similar methods. |
| `PageElement`           | Class | Abstract base class for `Tag` and `NavigableString`.           |
| `Formatter`             | Class | Handles output formatting like whitespace and indentation.     |

---

### Submodules (Internal Use – Not Usually Imported Directly)

> These exist within the `bs4` package but are typically used internally.

| Submodule       | Description                                                                        |
| --------------- | ---------------------------------------------------------------------------------- |
| `bs4.builder`   | Contains parser-specific builder classes to build the parse tree.                  |
| `bs4.element`   | Defines core classes like `Tag`, `NavigableString`, `Comment`, etc.                |
| `bs4.dammit`    | Handles Unicode decoding and fixing badly encoded documents (via `UnicodeDammit`). |
| `bs4.formatter` | Used for controlling how text is formatted when converting back to string.         |

---

### Parser Builders (Inside `bs4.builder`)

| Builder Class           | Backend Used                    |
| ----------------------- | ------------------------------- |
| `HTMLParserTreeBuilder` | Python’s built-in `html.parser` |
| `LXMLTreeBuilder`       | `lxml` HTML parser              |
| `LXMLTreeBuilderForXML` | `lxml` XML parser               |
| `HTML5TreeBuilder`      | `html5lib` parser               |

---

### Utility Components

| Component                           | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `UnicodeDammit` (from `bs4.dammit`) | Automatically detects and converts character encodings. |
| `XMLParsedAsHTMLWarning`            | Warning if XML is parsed with an HTML parser.           |
| `FeatureNotFound`                   | Exception raised when a requested parser is not found.  |

---

### Dependency Structure Overview

```plaintext
bs4/
├── __init__.py
├── builder/
│   ├── __init__.py
│   ├── _htmlparser.py
│   ├── _lxml.py
│   └── _html5lib.py
├── dammit.py
├── element.py
├── formatter.py
```

---
