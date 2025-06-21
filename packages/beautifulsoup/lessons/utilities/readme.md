## **Utilities in BeautifulSoup**

BeautifulSoup provides several utility functions and methods that help simplify the process of parsing, navigating, and extracting data from HTML or XML documents. These utilities are designed to make your scraping tasks more efficient and provide additional functionality beyond basic parsing and searching. Below is a comprehensive overview of the utilities available in BeautifulSoup.

---

#### **1. `.prettify()`**

The `.prettify()` method is a utility that formats the parsed HTML or XML document into a more readable and indented version. This is useful for debugging or visual inspection of the structure of the document.

* **Usage**:

  * Returns a formatted version of the parsed document, making it easier to understand the document's structure.

  * **Example**:

    ```python
    html_doc = "<html><body><p>Sample Paragraph</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    print(soup.prettify())
    ```

    **Output**:

    ```html
    <html>
     <body>
      <p>
       Sample Paragraph
      </p>
     </body>
    </html>
    ```

---

#### **2. `.stripped_strings`**

The `.stripped_strings` property is a generator that yields all the strings from the parsed document, removing any leading or trailing whitespace.

* **Usage**:

  * Allows easy access to all the text content within the HTML, with excess whitespace removed.

  * **Example**:

    ```python
    html_doc = "<html><body><p>  Sample Paragraph  </p><p> Another Text  </p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    for string in soup.stripped_strings:
        print(repr(string))
    ```

    **Output**:

    ```python
    'Sample Paragraph'
    'Another Text'
    ```

---

#### **3. `.find_all()`**

The `.find_all()` method is one of the core utilities in BeautifulSoup, allowing you to search for all instances of a particular tag or attribute within a document.

* **Usage**:

  * Returns a list of all elements that match the given criteria (tag, class, id, etc.).

  * **Example**:

    ```python
    html_doc = "<html><body><p class='test'>Test Paragraph 1</p><p class='test'>Test Paragraph 2</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    paragraphs = soup.find_all('p', class_='test')
    for paragraph in paragraphs:
        print(paragraph)
    ```

    **Output**:

    ```html
    <p class="test">Test Paragraph 1</p>
    <p class="test">Test Paragraph 2</p>
    ```

---

#### **4. `.find()`**

The `.find()` method is similar to `.find_all()`, but it returns only the first match found. This is useful when you need a single element and want to avoid unnecessary searching.

* **Usage**:

  * Returns the first element that matches the given tag and attributes.

  * **Example**:

    ```python
    html_doc = "<html><body><p>First Paragraph</p><p>Second Paragraph</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    first_paragraph = soup.find('p')
    print(first_paragraph)
    ```

    **Output**:

    ```html
    <p>First Paragraph</p>
    ```

---

#### **5. `.get_text()`**

The `.get_text()` method extracts all text from the parsed HTML or XML document, effectively flattening the structure and returning only the raw text content.

* **Usage**:

  * Useful for extracting the entire text from a document or specific element.

  * **Example**:

    ```python
    html_doc = "<html><body><p>First Paragraph</p><p>Second Paragraph</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    text = soup.get_text()
    print(text)
    ```

    **Output**:

    ```text
    First Paragraph
    Second Paragraph
    ```

---

#### **6. `.get()`**

The `.get()` method is used to retrieve the value of a specific attribute from an element. It is similar to accessing attributes directly, but provides a safer way by returning `None` if the attribute does not exist, instead of throwing an error.

* **Usage**:

  * Retrieve an attribute value from an element. Can be used with tags and attributes.

  * **Example**:

    ```python
    html_doc = "<html><body><a href='http://example.com'>Example Link</a></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    link = soup.find('a')
    href = link.get('href')
    print(href)
    ```

    **Output**:

    ```text
    http://example.com
    ```

---

#### **7. `.attrs`**

The `.attrs` property returns a dictionary of all the attributes of a tag. This is useful when you need to access or manipulate multiple attributes of an element.

* **Usage**:

  * Returns a dictionary of attributes for a given element.

  * **Example**:

    ```python
    html_doc = "<html><body><a href='http://example.com' class='link'>Example Link</a></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    link = soup.find('a')
    attributes = link.attrs
    print(attributes)
    ```

    **Output**:

    ```python
    {'href': 'http://example.com', 'class': ['link']}
    ```

---

#### **8. `.decompose()`**

The `.decompose()` method removes a tag and all of its contents from the document. This is useful when you want to clean up the parsed HTML by removing unwanted elements.

* **Usage**:

  * Removes an element and its content from the parse tree entirely.

  * **Example**:

    ```python
    html_doc = "<html><body><p>Text to keep</p><p>Text to remove</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    unwanted = soup.find('p', text="Text to remove")
    unwanted.decompose()  # Removes the unwanted <p> tag
    print(soup)
    ```

    **Output**:

    ```html
    <html><body><p>Text to keep</p></body></html>
    ```

---

#### **9. `.insert_before()` and `.insert_after()`**

These methods allow you to insert new elements or text into the document, either before or after a specified element. They are useful for modifying the document structure dynamically.

* **Usage**:

  * `insert_before()`: Inserts a tag before another element.

  * `insert_after()`: Inserts a tag after another element.

  * **Example**:

    ```python
    html_doc = "<html><body><p>First Paragraph</p><p>Second Paragraph</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    new_tag = soup.new_tag("p")
    new_tag.string = "Inserted Paragraph"
    soup.find('p').insert_after(new_tag)
    print(soup)
    ```

    **Output**:

    ```html
    <html><body><p>First Paragraph</p><p>Second Paragraph</p><p>Inserted Paragraph</p></body></html>
    ```

---

#### **10. `.extract()`**

The `.extract()` method is used to remove a tag from the document and return it. It is similar to `.decompose()`, but `.extract()` keeps the element in memory and can be reused.

* **Usage**:

  * Removes a tag from the document but keeps it for later use.

  * **Example**:

    ```python
    html_doc = "<html><body><p>Text to keep</p><p>Text to remove</p></body></html>"
    soup = BeautifulSoup(html_doc, 'html.parser')
    unwanted = soup.find('p', text="Text to remove")
    extracted = unwanted.extract()
    print(soup)
    print(extracted)
    ```

    **Output**:

    ```html
    <html><body><p>Text to keep</p></body></html>
    <p>Text to remove</p>
    ```

---

### Summary of Utilities in BeautifulSoup:

* **`.prettify()`**: Formats the document for better readability.
* **`.stripped_strings`**: Generates a sequence of strings with whitespace removed.
* **`.find_all()`**: Finds all matching elements in the document.
* **`.find()`**: Finds the first matching element.
* **`.get_text()`**: Extracts all text content from the document.
* **`.get()`**: Safely retrieves an attribute's value.
* **`.attrs`**: Accesses the attributes of a tag as a dictionary.
* **`.decompose()`**: Completely removes an element and its content from the document.
* **`.insert_before()` and `.insert_after()`**: Insert new elements before or after a specified element.
* **`.extract()`**: Removes an element from the document and returns it.

These utilities can significantly enhance your ability to parse, manipulate, and clean data from HTML or XML documents in BeautifulSoup.
