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
import os
from typing import Any
from zoneinfo import ZoneInfo
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import google_search
from google.adk.workflow import node
from google.genai import types

# Setup Google Cloud project credentials and environment variables
try:
    _, project_id = google.auth.default()
except Exception:
    project_id = None

project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "demo-project")
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
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
        query: The name of the city to get the current time for.

    Returns:
        A string with the current time information, or an error message with explicit recovery instructions.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return (
            f"Error: Timezone information is unavailable for query '{query}'. "
            "Currently, timezone info is only available for 'San Francisco' or 'SF'. "
            "Please ask the user to clarify or specify a supported location (such as San Francisco)."
        )

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


# ==========================================
# Pydantic Schemas for Multi-Agent Task Flows
# ==========================================


class SearchInput(BaseModel):
    query: str = Field(description="The search query to execute on Google.")


class SearchOutput(BaseModel):
    results_summary: str = Field(
        description="A clean, concise summary of the search results found."
    )


class ResearchInput(BaseModel):
    product_name: str = Field(
        description="The confirmed, unambiguous product name to research."
    )


class ResearchOutput(BaseModel):
    product_name: str = Field(description="Exact name of the product.")
    price_range: str = Field(description="The price range or MSRP of the product.")
    key_features: list[str] = Field(
        description="List of key features, specifications, and hardware details."
    )
    ratings_summary: str = Field(
        description="Summary of user and expert reviews, ratings, and feedback."
    )
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
    products: list[ProductInfo] = Field(
        description="List of structured research data for the confirmed products to compare."
    )


# ==========================================
# Specialized Standalone Sub-Agents Definition
# ==========================================

search_helper_agent = Agent(
    name="search_helper_agent",
    description="Specialized helper to perform Google Web Search and return a summarized set of results or potential product matches.",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="single_turn",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction="""You are the Search Helper Agent. Your specialized task is to perform Google Search queries and disambiguate input product names on behalf of the Coordinator.

You have access to:
- `google_search`: A tool to search Google.

Guidelines:
1. Parse the user request to identify specific product names or models.
2. Use `google_search` to confirm product availability and details or resolve ambiguities.
3. Return ONLY a comma-separated list of validated product names (for example: "iPhone 15 Pro, Google Pixel 8 Pro"). Do not include any sentences, introductory text, explanations, or markdown formatting.""",
    tools=[google_search],
)


research_agent = Agent(
    name="research_agent",
    description="Researches a single confirmed product to extract detailed specifications, pricing, reviews, pros, and cons.",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="single_turn",
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
    mode="single_turn",
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


# ==========================================
# Dynamic Workflow Node (ADK 2.0)
# ==========================================


@node(rerun_on_resume=True)
async def product_comparator_workflow(ctx: Context, node_input: Any = None) -> str:
    """Dynamic workflow orchestrating intake, search disambiguation, sequential research, and synthesis comparison."""

    user_query = node_input
    if not user_query and ctx.user_content and ctx.user_content.parts:
        parts_text = [p.text for p in ctx.user_content.parts if isinstance(p.text, str)]
        user_query = " ".join(parts_text)

    if not user_query:
        user_query = "Compare products requested by user."

    # Await search_helper_agent (mode: task) to extract and validate products
    search_response = await ctx.run_node(
        search_helper_agent,
        node_input=user_query,
    )

    # Extract list of products from search_response
    if isinstance(search_response, list):
        validated_products = search_response
    elif isinstance(search_response, str):
        cleaned = search_response.strip()
        if "," in cleaned:
            validated_products = [p.strip() for p in cleaned.split(",") if p.strip()]
        else:
            validated_products = [
                line.strip() for line in cleaned.splitlines() if line.strip()
            ]
    else:
        validated_products = [str(search_response)]

    if not validated_products:
        validated_products = [user_query]

    # 2. Sequential Research Loop
    research_collection = []
    for product in validated_products:
        product_research = await ctx.run_node(
            research_agent,
            node_input=f"Research product: {product}",
        )
        research_collection.append(str(product_research))

    # 3. Synthesis
    aggregated_research = "\n\n---\n\n".join(research_collection)
    comparison_input = (
        f"Compare the following researched products:\n\n{aggregated_research}"
    )

    comparison_result = await ctx.run_node(
        comparison_agent,
        node_input=comparison_input,
    )

    # 4. Return
    return str(comparison_result)


root_agent = product_comparator_workflow


app = App(
    root_agent=product_comparator_workflow,
    name="app",
)
