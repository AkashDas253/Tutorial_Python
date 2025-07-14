## BeautifulSoup Cheatsheet

---

#### **1. Creating a Soup Object**

* **Parse a string of HTML**:

  ```python
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(html_string, 'html.parser')
  ```

* **Parse from a file**:

  ```python
  with open('file.html', 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
  ```

---

#### **2. Navigating the Parse Tree**

* **Access the document’s title**:

  ```python
  soup.title
  ```

* **Get the title’s text**:

  ```python
  soup.title.string
  ```

* **Access the parent of an element**:

  ```python
  soup.title.parent
  ```

* **Access the first child**:

  ```python
  soup.body.contents[0]
  ```

* **Iterate over all children**:

  ```python
  for child in soup.body.children:
      print(child)
  ```

* **Access all descendants of an element**:

  ```python
  for descendant in soup.body.descendants:
      print(descendant)
  ```

---

#### **3. Searching the Tree**

* **Find a tag by name**:

  ```python
  tag = soup.find('tagname')
  ```

* **Find all tags by name**:

  ```python
  tags = soup.find_all('tagname')
  ```

* **Search by multiple attributes**:

  ```python
  soup.find_all('tagname', {'class': 'classname', 'id': 'idname'})
  ```

* **Search by CSS selector**:

  ```python
  soup.select('div.classname')
  ```

* **Find a tag by a specific attribute**:

  ```python
  soup.find('tagname', {'href': 'value'})
  ```

---

#### **4. Working with Attributes**

* **Access attributes of a tag**:

  ```python
  tag['attribute_name']
  ```

* **Access using `get()`**:

  ```python
  tag.get('attribute_name')
  ```

* **Modify an attribute**:

  ```python
  tag['attribute_name'] = 'new_value'
  ```

* **Add a new attribute**:

  ```python
  tag['new_attribute'] = 'value'
  ```

* **Remove an attribute**:

  ```python
  del tag['attribute_name']
  ```

* **Remove all attributes**:

  ```python
  tag.attrs.clear()
  ```

* **Search tags by attribute**:

  ```python
  soup.find_all('tagname', {'class': 'classname'})
  ```

---

#### **5. Working with Strings**

* **Get text from a tag**:

  ```python
  tag.string
  ```

* **Get all text from a tag**:

  ```python
  tag.get_text()
  ```

* **Strip extra whitespace**:

  ```python
  tag.get_text(strip=True)
  ```

* **Extract the value of a tag**:

  ```python
  tag.string
  ```

---

#### **6. Output**

* **Pretty-print HTML**:

  ```python
  print(soup.prettify())
  ```

* **Convert to string**:

  ```python
  str(soup)
  ```

---

#### **7. CSS Selectors**

* **Find tags using CSS selectors**:

  ```python
  soup.select('div.classname')
  ```

* **Find tags with a specific attribute using CSS selectors**:

  ```python
  soup.select('a[href="value"]')
  ```

* **Using CSS `nth-child` selector**:

  ```python
  soup.select('ul > li:nth-child(2)')
  ```

---

#### **8. Advanced Usage**

* **Navigating sibling elements**:

  ```python
  tag.find_next_sibling()  # Next sibling
  tag.find_previous_sibling()  # Previous sibling
  ```

* **Searching by regular expression**:

  ```python
  import re
  soup.find_all('a', href=re.compile('^http'))
  ```

* **Accessing specific index in a result**:

  ```python
  soup.find_all('tagname')[0]
  ```

* **Recursive search with `find_all()`**:

  ```python
  soup.find_all('tagname', recursive=True)
  ```

---

#### **9. Utilities**

* **Get the HTML of a tag**:

  ```python
  str(tag)
  ```

* **Get the raw HTML of the tag** (including any unparsed entities):

  ```python
  tag.encode_contents()
  ```

* **Find all elements by class**:

  ```python
  soup.find_all(class_='class_name')
  ```

* **Find all elements by ID**:

  ```python
  soup.find_all(id='id_name')
  ```

---

### **General Tips**

* **Handling Unicode**:

  ```python
  print(soup.prettify())
  ```

* **Dealing with broken HTML**:
  BeautifulSoup automatically deals with badly formed HTML and attempts to parse it.

* **Install BeautifulSoup**:

  ```bash
  pip install beautifulsoup4
  ```

---

### Summary

* **Navigating the Tree**: Use `.parent`, `.children`, `.descendants` for traversing HTML structure.
* **Searching**: `.find()`, `.find_all()`, and `.select()` help locate tags based on criteria.
* **Attributes**: Use `.attrs`, `.get()`, and `.get_text()` for working with tag attributes.
* **CSS Selectors**: `.select()` enables searching via CSS selectors.
* **Output**: `.prettify()` and `.string` for better output formatting.
