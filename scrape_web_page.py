from bs4 import BeautifulSoup
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
elif not api_key.startswith("sk-"):
    raise ValueError("Invalid API key format. Please ensure it starts with 'sk-'.")
else:
    print("API key loaded successfully.")

openai = OpenAI()

def count_num_of_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    model_to_encoding = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",  # If you're using custom naming
    }
    encoding_name = model_to_encoding.get(model, "cl100k_base")  # fallback to cl100k_base
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def scrape_web_page(url):
    """
    Scrape the content of a web page given its URL.

    Args:
        url (str): The URL of the web page to scrape.

    Returns:
        str: The text content of the web page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return ""

def summarize_content(content:str) -> str:
    """
    Summarize the content of a web page.

    Args:
        content (str): The text content of the web page.

    Returns:
        str: A summary of the content.
    """
    messages = [
        {
            "role" : "system",
            "content" : "You are helpful assistant that takes a long text and can summarize it in a very concise and precise manner that will help the user understand what this text is all about. \n"
            "You will not add any unwanted or unnecessary information to the summary."
        },
        {
            "role" : "user",
            "content" : f"\n Summarize the following content: {content} in markdown format."
        }
    ]

    chat_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    return chat_response.choices[0].message.content

def get_available_encodings():
    """
    Get the available encoding models from OpenAI.

    Returns:
        list: A list of available encoding models.
    """
    return tiktoken.list_encoding_names()

if __name__ == "__main__":
    # Example usage
    print("Available encoding models:", get_available_encodings())

    website_request = input("Enter the URL (Ex: http://www.example.com/) of the website to scrape: ")
    content = scrape_web_page(website_request)
    if content:
        token_count = count_num_of_tokens(content)
        print(f"Number of tokens in the content: {token_count}")
        summary = summarize_content(content)
        print("Summary of the content:")
        print(summary)
    else:
        print("No content scraped from the provided URL.")
    