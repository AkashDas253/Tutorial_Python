## BeautifulSoup Concepts and Subconcepts  

#### **1. Installation and Setup**  
- Installing BeautifulSoup (`pip install beautifulsoup4`)  
- Required dependencies (e.g., `lxml`, `html.parser`, `html5lib`)  

#### **2. Parsing HTML and XML**  
- `html.parser` (built-in)  
- `lxml` (faster, requires installation)  
- `html5lib` (most lenient, creates valid trees)  
- `xml` (for parsing XML)  

#### **3. Creating a Soup Object**  
- `BeautifulSoup(html_content, parser)`  
- Handling different encodings (`from_encoding`)  

#### **4. Navigating the Parse Tree**  
- **Tag Objects** (`soup.tag_name`)  
- **Navigating Up and Down**  
  - `.parent`, `.parents`  
  - `.contents`, `.children`  
- **Navigating Sideways**  
  - `.next_sibling`, `.previous_sibling`  
  - `.next_siblings`, `.previous_siblings`  

#### **5. Searching the Parse Tree**  
- **Find Methods**  
  - `find(name, attrs, recursive, text, limit, **kwargs)`  
  - `find_all(name, attrs, recursive, text, limit, **kwargs)`  
  - `find_parent()`, `find_parents()`  
  - `find_next()`, `find_all_next()`  
  - `find_previous()`, `find_all_previous()`  
- **CSS Selectors**  
  - `select(selector)`, `select_one(selector)`  

#### **6. Modifying the Parse Tree**  
- **Changing Tag Attributes** (`tag['attribute'] = value`)  
- **Modifying Tag Content** (`tag.string = new_value`)  
- **Appending and Inserting Elements**  
  - `.append()`, `.insert()`  
  - `.extend()`  
- **Replacing and Deleting Elements**  
  - `.replace_with(new_element)`  
  - `.decompose()`, `.extract()`  

#### **7. Getting Data from Tags**  
- `.text` (extracts all text)  
- `.string` (extracts direct text)  
- `.get_text(separator, strip)`  

#### **8. Handling Attributes**  
- `.attrs` (dictionary of attributes)  
- `tag['attribute']` (accessing specific attributes)  
- `.get('attribute', default_value)`  

#### **9. Encoding and Output Formatting**  
- `.prettify()` (pretty-printing HTML)  
- `.encode()`, `.decode()`  

#### **10. Handling NavigableString Objects**  
- `NavigableString` (text inside a tag)  
- `tag.string.replace_with(new_string)`  

#### **11. Working with Comments**  
- `Comment` object (`from bs4 import Comment`)  
- Extracting comments (`isinstance(tag, Comment)`)  

#### **12. Handling Malformed HTML**  
- BeautifulSoup's auto-fix feature  
- Choosing `html5lib` for strict correction  

#### **13. Extracting Links and Images**  
- Extracting URLs (`soup.find_all('a')['href']`)  
- Extracting image sources (`soup.find_all('img')['src']`)  

#### **14. Working with Tables**  
- Extracting table rows (`find_all('tr')`)  
- Extracting table data (`find_all('td')`)  

#### **15. Combining with Requests for Web Scraping**  
- Fetching HTML with `requests`  
- Parsing the fetched content  

#### **16. Handling JavaScript-Rendered Content**  
- Using Selenium or requests-html  
- Extracting dynamically loaded data  
