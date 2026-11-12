# Restaurant order manager

This project uses LangGraph to manage restaurant orders for Burger, Pizza,
Momos, and Malai_chaap. It supports partial availability, three order
attempts, two cooking attempts, and two serving attempts. Cooking and serving
fail with a 40% probability.

Run it from this directory with:

```powershell
uv run restproject
```

Set `GROQ_API_KEY` in `.env` to use Groq's `openai/gpt-oss-120b` structured
order extraction. Without a key, the included deterministic parser supports
the menu and simple yes/no partial-order replies.