import os
from groq import Groq

client = Groq(
    api_key="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",
)
def generate_llama(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content
