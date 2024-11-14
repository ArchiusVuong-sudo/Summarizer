import requests
from dotenv import find_dotenv, dotenv_values
from unstructured_client import UnstructuredClient
from langchain_unstructured import UnstructuredLoader
from langchain_community.callbacks import get_openai_callback
from unstructured_client.utils import BackoffStrategy, RetryConfig
from unstructured.cleaners.core import (
    clean_extra_whitespace, 
    blank_line_grouper
)

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

class DocumentLoader():
    def __init__(self) -> None:
        self.config = dotenv_values(find_dotenv())
        self.client: UnstructuredClient = UnstructuredClient(
            client          = requests.Session(),
            api_key_auth    = self.config.get("UNSTRUCTURED_API_KEY"),
            server_url      = self.config.get("UNSTRUCTURED_API_ENDPOINT"),
            retry_config    = RetryConfig(
                strategy                = "backoff",
                retry_connection_errors = True,
                backoff                 = BackoffStrategy(
                    initial_interval    = 500,
                    max_interval        = 60000,
                    exponent            = 1.5,
                    max_elapsed_time    = 900000
                ),
            )
        )

    def load(self, file_path: str):
        loader = UnstructuredLoader(
            file_path                   = file_path,
            partition_via_api           = True,
            client                      = self.client,
            split_pdf_page              = True,
            strategy                    = 'hi_res',
            extract_image_block_types   = ["Image", "Table"],
            post_processors             = [
                        clean_extra_whitespace, 
                        blank_line_grouper
                    ]
        )

        docs = loader.load()

        llm = AzureChatOpenAI(
            azure_deployment = "gpt-4o-mini",
            api_version = "2024-08-01-preview"
        )

        for doc in docs:
            doc.metadata.pop('languages', None)
            if doc.metadata['category'] == 'Image':
                doc.metadata['category'] = 'NarrativeText'
                with get_openai_callback() as cb:
                    doc.page_content = llm.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": "Explain this image in detail"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{doc.metadata['image_base64']}"},
                                    },
                                ]
                            )
                        ]
                    ).content

                    print(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")

        return docs
