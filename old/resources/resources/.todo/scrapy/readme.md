# Scrapy Cheatsheet

## 1. Installing Scrapy
- pip install scrapy  # Install Scrapy

## 2. Creating a New Scrapy Project
- scrapy startproject project_name  # Create a new Scrapy project

## 3. Running the Spider
- scrapy crawl spider_name  # Run a spider

## 4. Creating a New Spider
- scrapy genspider spider_name domain.com  # Create a new spider
- class SpiderName(scrapy.Spider):
  - name = 'spider_name'  # Spider name
  - start_urls = ['http://example.com']  # Initial URLs

## 5. Parsing Responses
- def parse(self, response):
  - title = response.xpath('//title/text()').get()  # Extract data using XPath
  - yield {'title': title}  # Yield extracted data

## 6. Handling Pagination
- next_page = response.xpath('//a[@class="next"]/@href').get()  # Get next page URL
- yield response.follow(next_page, callback=self.parse)  # Follow pagination link

## 7. Storing Data
- scrapy crawl spider_name -o output.json  # Store data in JSON format
- scrapy crawl spider_name -o output.csv  # Store data in CSV format

## 8. Using Item Classes
- from scrapy import Item, Field  # Import Item and Field
- class MyItem(Item):
  - title = Field()  # Define fields for item

## 9. Pipelines
- ITEM_PIPELINES = {
  - 'project_name.pipelines.MyPipeline': 1,  # Configure item pipeline
}
- class MyPipeline:
  - def process_item(self, item, spider):
    - return item  # Process and return item

## 10. Middleware
- DOWNLOADER_MIDDLEWARES = {
  - 'project_name.middlewares.MyDownloaderMiddleware': 543,  # Configure downloader middleware
}

## 11. Settings
- DOWNLOAD_DELAY = 2  # Set download delay between requests
- USER_AGENT = 'my-user-agent'  # Set user agent

## 12. Debugging
- scrapy shell 'http://example.com'  # Open Scrapy shell for debugging
- response.xpath('//title/text()').get()  # Test XPath selectors in shell

## 13. Running Scrapy in a Different Environment
- scrapy crawl spider_name -s LOG_LEVEL=DEBUG  # Set log level to DEBUG
- scrapy crawl spider_name -s JOBDIR='job_directory'  # Resume interrupted jobs
