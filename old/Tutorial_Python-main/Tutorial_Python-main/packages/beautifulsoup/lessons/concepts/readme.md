## BeautifulSoup Concepts and Subconcepts

### Basics

* What is BeautifulSoup
* Installation (`pip install beautifulsoup4`)
* Importing (`from bs4 import BeautifulSoup`)
* Parser choices

  * `html.parser`
  * `lxml`
  * `html5lib`

### Creating a Soup Object

* From string
* From file
* Parser argument

### Navigating the Parse Tree

* `.tagname`
* `.contents`
* `.children`
* `.descendants`
* `.string`
* `.strings`
* `.stripped_strings`
* `.parent`
* `.parents`
* `.next_sibling`, `.previous_sibling`
* `.next_siblings`, `.previous_siblings`
* `.next_element`, `.previous_element`
* `.next_elements`, `.previous_elements`

### Searching the Tree

* `find(name, attrs, recursive, string, **kwargs)`
* `find_all(name, attrs, recursive, string, limit, **kwargs)`
* `find_parent(name, attrs, **kwargs)`
* `find_parents(name, attrs, **kwargs)`
* `find_next_sibling(name, attrs, **kwargs)`
* `find_next_siblings(name, attrs, **kwargs)`
* `find_previous_sibling(name, attrs, **kwargs)`
* `find_previous_siblings(name, attrs, **kwargs)`
* `find_next(name, attrs, **kwargs)`
* `find_all_next(name, attrs, **kwargs)`
* `find_previous(name, attrs, **kwargs)`
* `find_all_previous(name, attrs, **kwargs)`
* Search by tag name
* Search by attributes
* Search by class (`class_`)
* Search by id (`id`)
* Search by text (`string=`)
* Search using regular expressions
* Search using custom functions

### Modifying the Tree

* `.insert(position, new_element)`
* `.append(new_element)`
* `.extend([elements])`
* `.insert_before(new_element)`
* `.insert_after(new_element)`
* `.replace_with(new_element)`
* `.replace_with_children()`
* `.wrap(wrapper_tag)`
* `.unwrap()`
* `.extract()`
* `.decompose()`
* `.clear()`

### Attributes

* Access with `tag['attribute']`
* Get all attributes: `tag.attrs`
* Check if attribute exists
* Modify attributes
* Delete attributes

### Working with Strings

* `.string`
* `.get_text(strip=False, separator='')`
* `.text` (alias for `.get_text()`)

### Output

* `.prettify()`
* `str(soup)`
* `.encode()`
* `.decode()`

### CSS Selectors

* `select(selector)`
* `select_one(selector)`
* Supported selectors

  * Tag
  * Class (`.class`)
  * ID (`#id`)
  * Descendant (`tag1 tag2`)
  * Child (`tag1 > tag2`)
  * Attribute selector (`[attr=value]`)

### Advanced Usage

* Custom parsers
* Working with malformed HTML
* Integration with requests module
* NavigableString, Tag, Comment objects
* Handling namespaces
* XML support

### Utilities

* `BeautifulSoup.new_tag(name, attrs)`
* `BeautifulSoup.new_string(text)`
* `BeautifulSoup.decode_contents()`
* `BeautifulSoup.encode_contents()`
