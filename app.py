import json
import os
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# 1. Define tools
# -------------------------------------------------------------------

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
    }
    return movies.get(
        genre.strip().lower(),
        "No Indian movies found for that genre.",
    )


@tool
def convert_celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (temp_c * 1.8) + 32


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city in India."""
    try:
        geo_response = requests.get(
            "[geocoding-api.open-meteo.com](https://geocoding-api.open-meteo.com/v1/search)",
            params={
                "name": city.strip(),
                "count": 1,
                "language": "en",
                "format": "json",
                "countryCode": "IN",
            },
            timeout=10,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        results = geo_data.get("results", [])
        if not results:
            return f"Could not find an Indian city named '{city}'."

        location = results[0]

        weather_response = requests.get(
            "[api.open-meteo.com](https://api.open-meteo.com/v1/forecast)",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,weather_code",
                "temperature_unit": "celsius",
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current")
        if not current:
            return f"Current weather is unavailable for {location['name']}."

        result = {
            "resolved_city": location["name"],
            "state": location.get("admin1"),
            "country": location.get("country"),
            "temperature_celsius": current.get("temperature_2m"),
            "weather_code": current.get("weather_code"),
            "observation_time": current.get("time"),
        }
        return json.dumps(result)

    except requests.RequestException as exc:
        return f"Weather service request failed: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return f"Unexpected weather data: {exc}"


tools = [
    get_weather,
    search_movies,
    convert_celsius_to_fahrenheit,
]


# -------------------------------------------------------------------
# 2. Initialize the model and agent
# -------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your environment before "
        "starting the application."
    )

llm = ChatGoogleGenerativeAI(
    # gemma-4-31b-it does not appear to be a standard Gemini API model ID.
    # Replace this value if your account uses a different available model.
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a specialized assistant restricted to Indian weather "
        "and Indian cinema. Use tools when they are relevant. "
        "Do not answer questions about weather outside India. "
        "For every request outside Indian weather or Indian cinema, reply "
        "with exactly this sentence and nothing else: "
        "'I am not authorized to answer questions outside of Indian "
        "weather and cinema.'"
    ),
)


# -------------------------------------------------------------------
# 3. Define the LangServe input/output adapters
# -------------------------------------------------------------------

class AgentInput(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Message sent to the agent.",
    )


def format_for_agent(value: Any) -> dict:
    """Convert LangServe input into the agent message-state format."""
    if isinstance(value, AgentInput):
        user_input = value.input
    elif isinstance(value, dict):
        user_input = value["input"]
    else:
        raise TypeError("Input must be an AgentInput object or a dictionary.")

    return {
        "messages": [
            HumanMessage(content=user_input.strip()),
        ]
    }


def content_to_text(content: Any) -> str:
    """Convert text or structured message content into a string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])

        if text_parts:
            return "\n".join(text_parts)

    return json.dumps(content, ensure_ascii=False, default=str)


def extract_text_response(agent_output: Any) -> str:
    """Extract the final assistant message from the agent state."""
    if not isinstance(agent_output, dict):
        return content_to_text(agent_output)

    messages = agent_output.get("messages", [])

    # Search backward because the final item should be the assistant reply.
    for message in reversed(messages):
        message_type = getattr(message, "type", None)

        if message_type in {"ai", "assistant"}:
            return content_to_text(getattr(message, "content", ""))

    # Fallback in case the installed agent version returns nested state.
    for value in agent_output.values():
        if isinstance(value, dict) and "messages" in value:
            return extract_text_response(value)

    return content_to_text(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str,
)


# -------------------------------------------------------------------
# 4. Create the FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )

