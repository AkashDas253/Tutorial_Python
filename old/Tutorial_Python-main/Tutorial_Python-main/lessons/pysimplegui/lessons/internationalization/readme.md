## Internationalization in PySimpleGUI

Internationalization (i18n) in PySimpleGUI allows applications to support multiple languages, regional settings, and character encodings. Though PySimpleGUI does not have built-in i18n APIs, it fully supports internationalization through standard Python techniques.

---

### Key Concepts

| Concept               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Unicode Support        | All elements support Unicode text for multilingual characters              |
| External Translation   | Use `.po` files, JSON, or dictionaries to store language-specific strings  |
| Dynamic Language Swap  | Change displayed language at runtime by updating elements                  |
| Locale Formatting      | Use Python's `locale` module to adapt to regional formats (dates, numbers) |

---

### Unicode Text in Elements

```python
layout = [[sg.Text("こんにちは世界")], [sg.Button("送信")]]

window = sg.Window("国際化", layout)
```

- Supports Chinese, Japanese, Hindi, Arabic, Cyrillic, etc.
- Font issues may occur on platforms without international fonts—use custom fonts if needed.

---

### Using a Translation Dictionary

```python
translations = {
    "en": {"greet": "Hello", "submit": "Submit"},
    "fr": {"greet": "Bonjour", "submit": "Soumettre"}
}

lang = "fr"
layout = [[sg.Text(translations[lang]["greet"])], [sg.Button(translations[lang]["submit"])]]
```

To switch language dynamically, update the elements:

```python
window["-TEXT-"].update(translations["fr"]["greet"])
```

---

### Locale-Based Formatting

```python
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')  # Set French locale

formatted = locale.format_string("%.2f", 1234567.89, grouping=True)
```

Use this to format numbers, currencies, dates, etc., based on the user's region.

---

### Best Practices

- Keep translatable text in dictionaries or external files.
- Use keys in elements for runtime updates.
- Always test with multilingual input and interface labels.
- Include font fallbacks if the default does not support all languages.

---

### Limitations

- No built-in translation or locale management system
- UI directionality (e.g., right-to-left) is not automatically handled
- Complex pluralization rules and grammatical adjustments must be managed manually

---
