README

Project Description
This project extracts text data from a list of URLs, performs text analysis, and generates an Excel file with the analysis results. The analysis includes various text metrics such as positive and negative scores, polarity, subjectivity, and more.

Approach
1. Web Scraping: Extracted the title and main content of articles from the given URLs.
2. Text Cleaning: Removed stop words and non-alphabetic characters.
3. Text Analysis: Computed various text analysis metrics using TextBlob and NLTK.
4. Output: Saved the results in an Excel file with adjusted column widths for better visibility.

 How to Run the Script
1. Ensure you have Python installed on your system.
2. Install the required dependencies using the following command:
3. Place the `Input.xlsx` file in the same directory as the script.
4. Run the script using the following command:
5. The output will be saved in `Output Data Structure.xlsx`.

Dependencies
- pandas: Data manipulation and analysis library.
- requests: Library for making HTTP requests.
- beautifulsoup4: Library for parsing HTML and XML documents.
- textblob: Library for processing textual data.
- nltk: Natural Language Toolkit for working with human language data.
- aiohttp: Asynchronous HTTP client/server framework.
- asyncio: Library for writing concurrent code using the async/await syntax.

Files Included in the Submission
- `python_script.py`: The main script to perform web scraping and text analysis.
- `Output Data Structure.xlsx`: The output file containing the analysis results.
- `README.txt`: This instruction file.
