import os
import weaviate

from crewai_tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Type, Optional
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth


class WeaviateToolSchema(BaseModel):
    """Input for WeaviateTool."""

    model_config = ConfigDict()
    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Weaviate database. Pass only the query, not the question.",
    )


class WeaviateTool(BaseTool):
    """Tool to search the Weaviate database"""

    name: str = "WeaviateTool"
    description: str = "A tool to search the Weaviate database for relevant information on internal documents"
    args_schema: Type[BaseModel] = WeaviateToolSchema
    query: Optional[str] = None

    def _run(self, query: str) -> str:
        """Search the Weaviate database

        Args:
            query (str): The query to search retrieve relevant information from the Weaviate database. Pass only the query as a string, not the question.

        Returns:
            str: The result of the search query
        """
        wcd_url = os.environ["WEAVIATE_API_ENDPOINT"]
        wcd_api_key = os.environ["WEAVIATE_API_KEY"]
        openai_key = os.environ["OPENAI_API_KEY"]
        if not openai_key:
            raise Exception("OPENAI_API_KEY is not set")

        headers = {"X-OpenAI-Api-Key": openai_key}

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=wcd_url,
            auth_credentials=Auth.api_key(wcd_api_key),
            headers=headers,
        )
        if not client.collections.exists("financial_docs"):
            client.collections.create(
                name="financial_docs",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                ),
                generative_config=Configure.Generative.openai(
                    model="gpt-4o-mini",
                ),
            )
        financial_docs = client.collections.get("financial_docs")
        # if you havent loaded the data yet, this can be used with the before_hook:

        # docs_to_load = os.listdir("knowledge")
        # with financial_docs.batch.dynamic() as batch:
        #     for d in docs_to_load:
        #         with open(os.path.join("knowledge", d), "r") as f:
        #             content = f.read()
        #         print("content", content)
        #         batch.add_object(
        #             {
        #                 "content": content,
        #                 "year": d.split("_")[0],
        #             }
        #         )
        response = financial_docs.generate.near_text(
            query=query,
            limit=5,
            grouped_task=f"Answer the question based on the provided query: {query}",
        )
        client.close()
        if response.generated:
            return response.generated
        else:
            return "No results found"
