# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import os
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import google_search, request_input
from google.genai import types

# Setup Google Cloud project credentials and environment variables
_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


# ==========================================
# Auxiliary Tools (Preserved for compatibility)
# ==========================================

def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


# ==========================================
# Pydantic Schemas for Multi-Agent Task Flows
# ==========================================

class SearchInput(BaseModel):
    query: str = Field(description="The search query to execute on Google.")


class SearchOutput(BaseModel):
    results_summary: str = Field(description="A clean, concise summary of the search results found.")


class ResearchInput(BaseModel):
    product_name: str = Field(description="The confirmed, unambiguous product name to research.")


class ResearchOutput(BaseModel):
    product_name: str = Field(description="Exact name of the product.")
    price_range: str = Field(description="The price range or MSRP of the product.")
    key_features: list[str] = Field(description="List of key features, specifications, and hardware details.")
    ratings_summary: str = Field(description="Summary of user and expert reviews, ratings, and feedback.")
    pros: list[str] = Field(description="List of key pros / positive aspects.")
    cons: list[str] = Field(description="List of key cons / negative aspects.")


class ProductInfo(BaseModel):
    product_name: str = Field(description="Exact name of the product.")
    price_range: str = Field(description="Price range or MSRP.")
    key_features: list[str] = Field(description="Key features and specs.")
    ratings_summary: str = Field(description="Review and rating summary.")
    pros: list[str] = Field(description="Pros of the product.")
    cons: list[str] = Field(description="Cons of the product.")


class ComparisonInput(BaseModel):
    products: list[ProductInfo] = Field(description="List of structured research data for the confirmed products to compare.")


# ==========================================
# Specialized Agents Definition
# ==========================================

search_helper_agent = Agent(
    name="search_helper_agent",
    description="Specialized helper to perform Google Web Search and return a summarized set of results or potential product matches.",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction="""You are the Search Helper Agent. Your specialized task is to perform Google Search queries on behalf of the Coordinator.

You have access to:
- `google_search`: A tool to search Google.

Guidelines:
1. Use `google_search` to find relevant and accurate search results for the given query.
2. Return a clean, summarized description of the results, specifically highlighting potential specific product models, configurations, or candidates if the query is a product name.""",
    tools=[google_search],
)


research_agent = Agent(
    name="research_agent",
    description="Researches a single confirmed product to extract detailed specifications, pricing, reviews, pros, and cons.",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction="""You are the Research Agent. Your specialized task is to find detailed, accurate information about a single, confirmed product.

You have access to:
- `google_search`: A tool to search Google.

Guidelines:
1. Use `google_search` to find:
   - The current retail price range or estimated pricing for the product.
   - Key specifications and unique features.
   - Overall user and expert review sentiment, including typical ratings.
   - Distinct pros (strengths, positive aspects).
   - Distinct cons (weaknesses, negative aspects).
2. Synthesize and organize this information carefully.
3. Return a comprehensive and detailed text report containing all these attributes. Do not use placeholder values.""",
    tools=[google_search],
)


comparison_agent = Agent(
    name="comparison_agent",
    description="Compares multiple researched products and generates a synthesis summary and recommendation.",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="task",
    input_schema=ComparisonInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction="""You are the Comparison Agent. Your task is to receive structured data for multiple confirmed products and synthesize them to provide a comprehensive, objective, and clear comparison.

Guidelines:
1. Compare the products on key dimensions (e.g., price, specs, pros/cons, review sentiments).
2. Generate:
   - A clear, structured comparative Markdown table summarizing key specs and prices.
   - A synthesized summary of how they compare.
   - A balanced "Pros and Cons" comparison.
3. Offer tentative, unbiased guidance and tailored recommendation based on different user needs or priorities (e.g., "Best for battery life", "Best value", "Best for photography").""",
)


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are the Coordinator Agent for the "product_comparator" multi-agent system.
Your goal is to coordinate product comparison requests. You have access to:
- `search_helper_agent`: A specialized task sub-agent to run Google searches and retrieve summaries of likely candidates.
- `research_agent`: A specialized task sub-agent to research a single confirmed product.
- `comparison_agent`: A specialized task sub-agent to compare multiple researched products and generate a synthesis/recommendation.
- `request_input`: A tool to ask the user a question and wait for their response.
- `get_weather`, `get_current_time`: Auxiliary tools.

Workflow Guidelines:
1. Parse the user request to identify the specific list of product names or models they wish to compare (e.g., ["Google Pixel 8 Pro", "Samsung Galaxy S24 Ultra"]).
2. Validate and clarify each product name before proceeding to research:
   - For each input product, check if it is specific, unambiguous, and currently available by calling the `search_helper_agent` task with a search query.
   - If a product name is ambiguous (e.g., "iPhone 15" could mean standard, Plus, Pro, or Pro Max; or "Pixel 8" could mean standard or Pro), use `search_helper_agent` to find likely specific candidates.
   - If an input is ambiguous, call the `request_input` tool with a polite, clear message presenting the likely specific candidates and asking the user to confirm or choose which exact model they want to compare.
   - Wait for the user to confirm before researching that product.
3. Once all product names are fully validated and confirmed:
   - For each confirmed product, invoke the `research_agent` task sub-agent sequentially, passing the exact confirmed product name.
   - Keep track of the structured `research_agent` output for each product.
4. After gathering the research data for all confirmed products:
   - Call the `comparison_agent` task sub-agent with the compiled list of products' research data.
5. Present the comparison summary, comparative tables, pros and cons, and recommendations from `comparison_agent` to the user in a beautiful, premium, and professional markdown format. Use rich formatting and highlight key trade-offs.""",
    sub_agents=[search_helper_agent, research_agent, comparison_agent],
    tools=[get_weather, get_current_time, request_input],
)


app = App(
    root_agent=root_agent,
    name="app",
)
