import os
from openai import OpenAI

def test_connection():
    # make sure your key is exported:
    # export LLM_API_KEY="sk-..."
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise EnvironmentError("LLM_API_KEY not found. Run: export LLM_API_KEY='your-key'")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.302.ai/v1",   # or remove base_url if using OpenAI directly
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # or any model you’re testing
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello from the test script."}
            ],
            temperature=0.5,
        )
        print("✅ Connection successful!")
        print("Model reply:", response.choices[0].message.content)

    except Exception as e:
        print("❌ Connection failed.")
        print(e)

if __name__ == "__main__":
    test_connection()
