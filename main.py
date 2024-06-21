import os
import argparse

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from datasets import load_dataset

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


MODEL = "EleutherAI/gpt-neo-125m"

VST = TypeVar("VST", bound="VectorStore")


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


# original BM25Retriever.from_documents
# @classmethod
# def from_documents(
#     cls,
#     documents: Iterable[Document],
#     *,
#     bm25_params: Optional[Dict[str, Any]] = None,
#     preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
#     **kwargs: Any,
# ) -> BM25Retriever:
#     """
#     Create a BM25Retriever from a list of Documents.
#     Args:
#         documents: A list of Documents to vectorize.
#         bm25_params: Parameters to pass to the BM25 vectorizer.
#         preprocess_func: A function to preprocess each text before vectorization.
#         **kwargs: Any other arguments to pass to the retriever.

#     Returns:
#         A BM25Retriever instance.
#     """
#     texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
#     return cls.from_texts(
#         texts=texts,
#         bm25_params=bm25_params,
#         metadatas=metadatas,
#         preprocess_func=preprocess_func,
#         **kwargs,
#     )


# generator BM25Retriever.from_documents
# @classmethod
# def from_documents(
#     cls,
#     documents: Iterable[Document],
#     *,
#     bm25_params: Optional[Dict[str, Any]] = None,
#     preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
#     **kwargs: Any,
# ):
#     """
#     Create a BM25Retriever from a list of Documents.
#     Args:
#         documents: A list of Documents to vectorize.
#         bm25_params: Parameters to pass to the BM25 vectorizer.
#         preprocess_func: A function to preprocess each text before vectorization.
#         **kwargs: Any other arguments to pass to the retriever.

#     Returns:
#         A BM25Retriever instance.
#     """

#     def generator():
#         for d in documents:
#             yield (d.page_content, d.metadata)

#     texts, metadatas = zip(*generator())
#     return cls.from_texts(
#         texts=texts,
#         bm25_params=bm25_params,
#         metadatas=metadatas,
#         preprocess_func=preprocess_func,
#         **kwargs,
#     )


# orignal VST.from_documents
# @classmethod
# def from_documents(
#     cls: Type[VST],
#     documents: List[Document],
#     embedding: Embeddings,
#     **kwargs: Any,
# ) -> VST:
#     """Return VectorStore initialized from documents and embeddings.

#     Args:
#         documents: List of Documents to add to the vectorstore.
#         embedding: Embedding function to use.
#     """
#     texts = [d.page_content for d in documents]
#     metadatas = [d.metadata for d in documents]
#     return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)


# generator VST.from_documents
# @classmethod
# def from_documents(
#     cls: Type[VST],
#     documents: List[Document],
#     embedding: Embeddings,
#     **kwargs: Any,
# ) -> VST:
#     """Return VectorStore initialized from documents and embeddings.

#     Args:
#         documents: List of Documents to add to the vectorstore.
#         embedding: Embedding function to use.
#     """

#     def generator():
#         for d in documents:
#             yield (d.page_content, d.metadata)

#     texts, metadatas = zip(*generator())
#     # texts = [d.page_content for d in documents]
#     # metadatas = [d.metadata for d in documents]
#     return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)


# def text_split(dataset):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=100,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     for example in dataset:
#         all_text = f"title: {example['title']}\ntext: {example['text']}"
#         split_texts = text_splitter.create_documents([all_text])
#         for split_text in split_texts:
#             yield split_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ret", default="BM25", type=str, choices=["BM25", "ElasticSearchBM25", "DPR"]
    )
    parser.add_argument("--db", default="", type=str)
    args = parser.parse_args()

    dataset = load_dataset("heegyu/namuwiki", split="train[:1%]")  # for test
    # dataset = load_dataset("heegyu/namuwiki", split="train", streaming=True)

    def make_all_text(example):
        return {"all_text": f"title: {example['title']}\ntext: {example['text']}"}

    dataset = dataset.map(make_all_text, batched=False)

    # generator
    # texts = text_split(dataset)

    # no generator
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = splitter.create_documents(dataset["all_text"])

    if args.ret == "BM25":
        retriever = BM25Retriever.from_documents(texts)
        retriever.k = 3

    if args.ret == "ElasticSearchBM25":
        raise ValueError("")

        retriever = ElasticSearchBM25Retriever.create(
            "http://localhost:9200", "langchain-index-4"
        )
        # retriever.k = 3 ???
        retriever.add_texts(texts)
        # https://python.langchain.com/v0.2/docs/integrations/retrievers/elastic_search_bm25/
    elif args.ret == "DPR":
        # [all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2]
        # https://huggingface.co/sentence-transformers

        path = os.path.join("db", args.db)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": False},
        )

        if os.path.isdir(path):
            print("load from local")
            db = FAISS.load_local("faiss_index", embeddings)
        else:
            db = FAISS.from_documents(texts, embeddings)

            db.save_local(path)

        # print(db.index.ntotal)

        retriever = db.as_retriever(search_kwargs={"k": 1})

    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50},
    )

    system_prompt = "질문에 답변해줘" "\n\n" "{context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "나무위키가 뭐야?"})
    print(response)
