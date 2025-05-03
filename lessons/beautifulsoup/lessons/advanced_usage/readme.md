## **Advanced Usage in BeautifulSoup**

---

#### **1. Navigating Sibling Elements**

* **Find the next sibling element**:

  ```python
  tag.find_next_sibling()
  ```

  * Retrieves the next sibling in the DOM tree of the current tag.

* **Find the previous sibling element**:

  ```python
  tag.find_previous_sibling()
  ```

  * Retrieves the previous sibling in the DOM tree.

* **Find all siblings** (including the current tag):

  ```python
  tag.find_all_next()
  tag.find_all_previous()
  ```

---

#### **2. Searching by Regular Expressions**

* **Search using regular expressions**:

  * BeautifulSoup supports searching for elements with attributes that match regular expressions.

  ```python
  import re
  soup.find_all('a', href=re.compile('^http'))
  ```

  * The `re.compile()` function allows patterns like `^http` to find all links starting with "http".

* **Search within text using regex**:

  ```python
  soup.find_all(text=re.compile('example'))
  ```

  * Find all elements containing the text "example" using regular expressions.

---

#### **3. Accessing Specific Index in Result**

* **Access the nth tag in the result set**:

  ```python
  tag = soup.find_all('a')[0]
  ```

  * When you have a list of elements returned by `.find_all()`, you can access a specific element using its index (starting at 0).

---

#### **4. Recursive Search**

* **Search within nested elements** (default is recursive):

  ```python
  soup.find_all('tagname', recursive=True)
  ```

  * By default, `.find_all()` searches the entire subtree (all descendants).
* **Search only immediate children (non-recursive)**:

  ```python
  soup.find_all('tagname', recursive=False)
  ```

  * Limits the search to the immediate children of the tag, excluding nested tags.

---

#### **5. Working with Non-HTML Content**

* **Extract JavaScript, CSS, or other non-HTML content**:

  * BeautifulSoup allows extraction of other types of content embedded within HTML.

  ```python
  script_tags = soup.find_all('script')
  ```

  * This extracts `<script>` tags, which can contain JavaScript.

---

#### **6. Using `find_parent()` and `find_all_parents()`**

* **Find the parent tag**:

  ```python
  parent = tag.find_parent()
  ```

  * Returns the closest parent tag of the current element.

* **Find all parent tags**:

  ```python
  parents = tag.find_all_parents()
  ```

  * Returns a list of all parents of the current element in the DOM hierarchy.

---

#### **7. Navigating Through Tags with `find_next()` and `find_previous()`**

* **Find the next tag after the current tag**:

  ```python
  tag.find_next()
  ```

  * Finds the next tag in the document, skipping over tags that are not elements (e.g., navigates to the next tag).

* **Find the previous tag before the current tag**:

  ```python
  tag.find_previous()
  ```

  * Finds the previous tag, again skipping over non-tag elements.

* **Find the next sibling that matches a filter**:

  ```python
  tag.find_next_sibling('tagname')
  ```

---

#### **8. Modifying the Document Tree**

* **Insert a new tag**:

  ```python
  new_tag = soup.new_tag('a', href='https://example.com')
  tag.insert_before(new_tag)  # Insert before the current tag
  tag.insert_after(new_tag)   # Insert after the current tag
  ```

* **Insert text**:

  ```python
  tag.insert_before("Some text")
  ```

* **Replace a tag with new content**:

  ```python
  tag.replace_with('New content')
  ```

---

#### **9. Handling Non-Standard HTML**

* **Handle malformed HTML**:
  BeautifulSoup can still parse malformed HTML and fix issues, such as missing closing tags, improperly nested elements, etc. For example, `<p><b>Test</b>` will be corrected to `<p><b>Test</b></p>`.

---

#### **10. Extracting Non-HTML Content**

* **Extract raw content of a tag** (e.g., image `src` or anchor `href`):

  ```python
  src = tag['src']
  href = tag['href']
  ```

* **Extract non-HTML content like JavaScript**:

  ```python
  scripts = soup.find_all('script')
  for script in scripts:
      print(script.string)
  ```

---

#### **11. Performance Considerations**

* **Limiting the search scope**:

  * Limit the search for specific sections using `find()` within a portion of the document:

  ```python
  section = soup.find('div', class_='content')
  links = section.find_all('a')
  ```

* **Using `lxml` parser** for faster performance:

  * BeautifulSoup can be configured to use `lxml`, which is faster than the default `html.parser`.

  ```python
  soup = BeautifulSoup(html_string, 'lxml')
  ```

---

#### **12. Working with NavigableString and Comment**

* **Get raw text**: `NavigableString` represents text within tags.

  ```python
  text = tag.string
  ```

* **Work with HTML comments**:

  ```python
  comment = soup.find(string=lambda text: isinstance(text, Comment))
  ```

---

### Summary of Advanced Usage

* **Navigating siblings**: Use `find_next_sibling()` and `find_previous_sibling()` to navigate siblings.
* **Regex**: Search tags or text using `re.compile()`.
* **Indexing**: Access specific elements from search results by index.
* **Non-recursive search**: Limit search scope with `recursive=False`.
* **Manipulating content**: Insert, replace, and modify tags with `insert_before()`, `insert_after()`, and `replace_with()`.
* **Handle malformed HTML**: BeautifulSoup auto-corrects and parses broken HTML.
* **Performance**: Use the `lxml` parser for better speed.
