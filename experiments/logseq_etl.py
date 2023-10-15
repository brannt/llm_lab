from enum import Enum
from pathlib import Path
import sys
from typing import Any, Optional

import dotenv
import openai
import pydantic

from langchain.document_loaders.base import BaseLoader
from langchain.schema import BaseDocumentTransformer
from langchain.schema.vectorstore import VectorStore
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


class Settings(pydantic.BaseSettings):
    LOGSEQ_DIR: Path
    CHROMA_BASE_DIR: Path = pydantic.Field(..., env="CHROMA_DIR")
    CHROMA_PERSIST_DIR: Path = None
    OPENAI_API_KEY: str

    @pydantic.validator("CHROMA_PERSIST_DIR", pre=True)
    def set_chroma_persist_dir(cls, v: Path, values: dict) -> Path:
        return v or values["CHROMA_BASE_DIR"] / "logseq"


settings = None


def setup():
    dotenv.load_dotenv()
    global settings
    settings = Settings()
    openai.api_key = settings.OPENAI_API_KEY


class DocumentETL(pydantic.BaseModel):
    loader: BaseLoader
    transformer: BaseDocumentTransformer
    vector_store: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        docs = self.loader.load()
        transformed = self.transformer.transform_documents(docs)
        self.vector_store.add_documents(transformed)
        self.vector_store.persist()


def run_etl(etl: Optional[DocumentETL] = None) -> None:
    settings.CHROMA_PERSIST_DIR.unlink(missing_ok=True)
    settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    etl = etl or setup_etl()
    etl()


def setup_etl():
    logseq_location = settings.LOGSEQ_DIR

    loader = DirectoryLoader(
        logseq_location,
        glob="**/*.md",
        # FIXME: The Logseq data dir contains a 'logseq/bak' subdirectory that contains
        # old versions of the files. Need to add a param to DirectoryLoader to exclude this directory.
        # exclude_glob='logseq/bak/**/*.*',
        # https://github.com/langchain-ai/langchain/pull/11831
        loader_cls=UnstructuredMarkdownLoader,
        silent_errors=True,
    )
    chunk_size = 1000
    chunk_overlap = 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n- ", "\n", "\.", " ", ""],
    )
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=str(settings.CHROMA_PERSIST_DIR),
    )
    etl = DocumentETL(loader=loader, transformer=splitter, vector_store=vectordb)
    return etl


def run_query(query: str, qa_chain: Optional[RetrievalQA] = None) -> None:
    qa_chain = qa_chain or setup_query_chain()
    return qa_chain({"query": query})["result"]


def setup_query_chain():
    vectordb = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(settings.CHROMA_PERSIST_DIR),
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectordb.as_retriever(search_type="mmr")
    )

    return qa_chain


QUESTIONS = [
    "What potential projects do I have?",
    "How to train a model?",
    "Where is Iltar located?",
]


class Command(Enum):
    INGEST = "ingest"
    QUERY = "query"
    TEST = "test"


if __name__ == "__main__":
    setup()
    command = Command(sys.argv[1])
    if command is Command.INGEST:
        run_etl()
    elif command is Command.QUERY:
        query = sys.argv[2]
        print(run_query(query))
    elif command is Command.TEST:
        for question in QUESTIONS:
            print(f"Question: {question}")
            print(f"Answer: {run_query(question)}")
            print()
