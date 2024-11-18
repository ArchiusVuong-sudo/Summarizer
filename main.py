import os
import asyncio

import pandas as pd
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from core.metadata import Metadata
from core.document_summarizer import DocumentSummarizer


async def get_summary(llm: AzureChatOpenAI, docs: list[Document]) -> dict[str, any]:
    summarizer: DocumentSummarizer = DocumentSummarizer(llm, docs)
    
    app = summarizer.compile_app()

    step: dict = {}
    async for step in app.astream(
        {"contents": [doc.page_content for doc in docs]},
        {"recursion_limit": 10},
    ):
        pass

    return step

def extract_document(llm: AzureChatOpenAI, file_path: str):
    document_loader: PyPDFLoader = PyPDFLoader(file_path)
    docs: list[Document] = document_loader.load()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )

    runnable = prompt | llm.with_structured_output(schema = Metadata)

    summary = asyncio.run(get_summary(llm, docs))
    metadata = runnable.invoke({"text": summary})

    return metadata.subject, metadata.keywords, summary['generate_final_summary']['final_summary']

if __name__ == '__main__':
    load_dotenv()

    llm: AzureChatOpenAI = AzureChatOpenAI(
        model = "gpt-4o-mini",
        api_version = "2024-08-01-preview"
    )

    df = pd.DataFrame(columns=['file_name', 'subject', 'keywords', 'summary'])

    for file_name in os.listdir(os.environ['DOCUMENT_PATH']):
        file_path = os.path.join(os.environ['DOCUMENT_PATH'], file_name)
        if os.path.isfile(file_path):
            try:
                subject, keywords, summary = extract_document(llm, file_path)

                df = pd.concat([df, pd.DataFrame([{
                    'file_name': file_name, 
                    'subject': subject,
                    'keywords': keywords,
                    'summary': summary
                }])], ignore_index=True)
            except:
                print(f'- Cannot extract: {file_path}')

    df.to_excel("output_file.xlsx")  
