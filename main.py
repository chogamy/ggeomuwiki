from typing import Any, Callable, Dict, Iterable, List, Optional

from datasets import load_dataset

from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL = "EleutherAI/gpt-neo-125m"


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


@classmethod
def from_documents(
    cls,
    documents: Iterable[Document],
    *,
    bm25_params: Optional[Dict[str, Any]] = None,
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
    **kwargs: Any,
) -> BM25Retriever:
    """
    Create a BM25Retriever from a list of Documents.
    Args:
        documents: A list of Documents to vectorize.
        bm25_params: Parameters to pass to the BM25 vectorizer.
        preprocess_func: A function to preprocess each text before vectorization.
        **kwargs: Any other arguments to pass to the retriever.

    Returns:
        A BM25Retriever instance.
    """

    def generator():
        for d in documents:
            yield (d.page_content, d.metadata)

    texts, metadatas = zip(*generator())
    return cls.from_texts(
        texts=texts,
        bm25_params=bm25_params,
        metadatas=metadatas,
        preprocess_func=preprocess_func,
        **kwargs,
    )


def text_split(dataset):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    for example in dataset:
        all_text = f"title: {example['title']}\ntext: {example['text']}"
        split_texts = text_splitter.create_documents([all_text])
        for split_text in split_texts:
            yield split_text


if __name__ == "__main__":
    # dataset = load_dataset("heegyu/namuwiki", split="train[:1%]") # for test
    dataset = load_dataset("heegyu/namuwiki", split="train", streaming=True)

    def make_all_text(example):
        return {"all_text": f"title: {example['title']}\ntext: {example['text']}"}

    dataset = dataset.map(make_all_text, batched=False)

    texts = text_split(dataset)

    retriever = BM25Retriever.from_documents(texts)
    # retriever.k = 3

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
