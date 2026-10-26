"""Chat model factory: OpenAI or Azure OpenAI, chosen by LLM_PROVIDER in .env."""

import os

from dotenv import load_dotenv

from compliance_rag.ingest import PROJECT_ROOT

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def get_chat_model():
    load_dotenv(PROJECT_ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        # Endpoint, key and OPENAI_API_VERSION are read from the environment.
        return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
