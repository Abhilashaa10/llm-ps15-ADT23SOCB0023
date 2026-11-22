📚 Library Assistant Bot (PS15)
A chatbot that recommends books from a library catalogue CSV by topic or author.

How it works

The catalogue (catalogue.csv) is loaded with pandas.
The student's message is turned into keywords, and the catalogue is filtered to the matching books.
Only those matching rows are sent to the LLM, which recommends and explains the top 3 books (with shelf and availability).
If nothing matches, the LLM is not called, so the bot can't make up books.
Tech stack: Python, Streamlit, pandas, and an LLM through an OpenAI-compatible API (Groq Llama 3.3 by default; Gemini or Ollama also work).

Setup
pip install -r requirements.txt
Get a free API key from https://console.groq.com/keys
Copy .env.example to .env and paste your key in it.
Run the app:
streamlit run app.py
Files
File	Purpose
app.py	Streamlit chat UI
bot.py	Keyword filter, prompt, and LLM call
catalogue.csv	Sample catalogue of 50 books
.env.example	Template for API key settings
Example queries
Suggest books on machine learning
Any books by Andrew Tanenbaum?
I want to learn Python
Books about operating systems and memory management
Something on cooking (no match, so the bot says so)
