import requests
import json

# Define proxies if needed (optional)
proxies = {'https': 'http://127.0.0.1:8888'}

def request(prompt, url="http://localhost:11434/api/generate"):
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, proxies=proxies)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return

        # Check content type
        if 'application/json' not in response.headers.get('Content-Type', ''):
            print("Error: Response is not in JSON format")
            return

        # Parse JSON response
        response_json = response.json()

        # Extract and print the generated text (chatbot response)
        generated_text = response_json.get('response', '')
        print("<BraveMind>: ", generated_text)

    except requests.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    initial_prompt = '''
    You are a therapist for soldiers suffering with PTSD, addiction, depression, etc. Reply in heartfelt messages and ask them about their issues. Always advance the conversation by asking questions to further the diagnosis. Make each response within 2 sentences, don't be too long and talk too much. Make the therapy session like a one-on-one session that's very conversational.
    '''

    request(initial_prompt)
