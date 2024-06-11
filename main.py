from datasets import load_dataset


from langchain_huggingface import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL = "EleutherAI/gpt-neo-125m"


if __name__ == "__main__":
    dataset = load_dataset("heegyu/namuwiki", split="train[:10%]")

    # texts = [
    #     Document(page_content=f"title: {item['title']}\ntext: {item['text']}")
    #     for item in dataset
    # ]

    texts = [f"title: {item['title']}\ntext: {item['text']}" for item in dataset]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents(texts)

    retriever = BM25Retriever.from_documents(texts)

    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "top_k": 50,
            "temperature": 0.1,
        },
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
    print(response["answer"])
