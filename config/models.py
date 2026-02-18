from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()

mega_api_key = os.getenv("MEGALLM_API_KEY")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

model = ChatOpenAI(
    model = "qwen/qwen3-next-80b-a3b-instruct",
    base_url = "https://ai.megallm.io/v1",
    api_key = SecretStr(mega_api_key) ,
    rate_limiter = rate_limiter,
    request_timeout = 120,
    max_retries = 3,
)