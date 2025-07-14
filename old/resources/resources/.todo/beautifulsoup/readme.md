# Beautiful Soup Cheatsheet

## 1. Installing Beautiful Soup
- pip install beautifulsoup4  # Install Beautiful Soup
- pip install lxml  # Install lxml parser (optional)

## 2. Importing Libraries
- from bs4 import BeautifulSoup  # Import BeautifulSoup
- import requests  # Import requests for HTTP requests

## 3. Making a Request
- response = requests.get('http://example.com')  # Make a GET request
- html_content = response.content  # Get HTML content

## 4. Creating a Beautiful Soup Object
- soup = BeautifulSoup(html_content, 'lxml')  # Create Beautiful Soup object

## 5. Navigating the Parse Tree
- title = soup.title  # Access title tag
- title_text = soup.title.string  # Get title text
- first_paragraph = soup.find('p')  # Find first <p> tag

## 6. Finding Elements
- all_paragraphs = soup.find_all('p')  # Find all <p> tags
- specific_paragraph = soup.find('p', class_='class-name')  # Find <p> with specific class
- links = soup.find_all('a')  # Find all <a> tags

## 7. Accessing Attributes
- link = soup.find('a')
- link_url = link['href']  # Get href attribute
- img_src = soup.find('img')['src']  # Get src attribute of <img>

## 8. Modifying the Parse Tree
- new_tag = soup.new_tag('a', href='http://new-link.com')  # Create new tag
- soup.body.append(new_tag)  # Append new tag to <body>

## 9. Searching by CSS Selectors
- selected_items = soup.select('.class-name')  # Select by class
- selected_items = soup.select('div > p')  # Select <p> inside <div>

## 10. Extracting Text
- text_content = soup.get_text()  # Get all text content
- specific_text = soup.find('p').get_text(strip=True)  # Get text from specific tag

## 11. Handling Nested Elements
- nested_element = soup.find('div').find('span')  # Find nested <span> inside <div>

## 12. Pretty Printing
- print(soup.prettify())  # Print the HTML in a readable format

## 13. Encoding and Decoding
- response.encoding = 'utf-8'  # Set encoding
- html_content = response.content.decode('utf-8')  # Decode content
