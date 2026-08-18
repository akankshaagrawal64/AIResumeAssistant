import os

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy
)

from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI

from graph.nodes import retrieve, rerank

from evaluation.test_dataset import test_dataset


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# --------------------------------------------------
# LLM used by RAGAS
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=api_key
)

evaluator_llm = LangchainLLMWrapper(llm)


# --------------------------------------------------
# IMPORTANT
# --------------------------------------------------

namespace = "session-d1383df3-7c87-4d89-9129-78ea9e03b292"


# --------------------------------------------------
# Store evaluation data
# --------------------------------------------------

samples = []


# --------------------------------------------------
# Run tests
# --------------------------------------------------

for test in test_dataset:

    question = test["question"]

    ground_truth = test["ground_truth"]


    print("\n" + "=" * 60)

    print("Question:")
    print(question)


    # ==================================================
    # RETRIEVAL
    # ==================================================

    retrieve_result = retrieve({

        "question": question,

        "vectorstore_namespace": namespace
    })


    documents = retrieve_result["documents"]


    print(
        f"Retrieved documents: {len(documents)}"
    )


    # ==================================================
    # RERANKING
    # ==================================================

    rerank_result = rerank({

        "question": question,

        "documents": documents
    })


    reranked_documents = (
        rerank_result["reranked_documents"]
    )


    print(
        f"After reranking: "
        f"{len(reranked_documents)}"
    )


    # ==================================================
    # CONTEXT
    # ==================================================

    contexts = [

        doc.page_content

        for doc in reranked_documents
    ]


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    context_text = "\n\n".join(
        contexts
    )


    answer_prompt = f"""
Answer the question using ONLY the context below.

Question:
{question}

Context:
{context_text}
"""


    response = llm.invoke(
        answer_prompt
    )


    answer = response.content


    print("\nAnswer:")
    print(answer)


    # ==================================================
    # RAGAS SAMPLE
    # ==================================================

    samples.append({

        "user_input": question,

        "retrieved_contexts": contexts,

        "response": answer,

        "reference": ground_truth
    })


# --------------------------------------------------
# Create RAGAS dataset
# --------------------------------------------------

dataset = EvaluationDataset.from_list(
    samples
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

result = evaluate(

    dataset,

    metrics=[

        ContextRecall(),

        ContextPrecision(),

        Faithfulness(),

        AnswerRelevancy()
    ],

    llm=evaluator_llm
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n")
print("=" * 60)
print("RAG EVALUATION RESULTS")
print("=" * 60)

print(result)