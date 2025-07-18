# **Baseline System Report**

## **Analytic Approach**

### **Target Definition**

The primary goal of the baseline system is to generate financial insights and recommendations for users who ask investment-related 
questions. The system aims to deliver guidance that is: 
- Relevant to the user’s query.
- Factually accurate, relying on general financial domain knowledge.

#### Inputs
- **User Query:** A text prompt submitted by retail investors, advisors, or other end-users. These queries may be vague or incomplete, reflecting limited financial expertise.
- **Financial Knowledge Base (RAG based):** A basic collection of publicly available market data or static financial documents that can be used for reference. Those documents are collected from various sources (e.g alpaca) and stored in the Qdrant vector database, documents are fetched from the Qdrant vector database during the RAG process and selected based on the relevance to the user query.

#### What Kind of System Was Built?

The baseline system is a straightforward question-answering (QA) application using an open-source language model (falcon 7B). It accepts user queries and returns an answer drawn from the model’s internal knowledge and relevence data from the vector database.
It relies largely on the generic QA capabilities of the Falcon 7B model, providing quick answers but often lacking deep contextualization and faithfulness to user prompts.

## **System Description**

#### System Components
- **Open-Source LLM**: A small-scale language model (Falcon 7B) with domain-specific fine-tuning.
- **Simple Prompt**: User queries are fed directly into the model without structured or optimized prompts.
- **RAG based Knowledge Base**: Relevant documents are fetched from the Qdrant vector database during the RAG process and selected based on the relevance to the user query.

#### Data Flow (Baseline)

- User submits an investment-related query
- Query is processed to fetch relevant documents from Qdrant vector database
- Falcon 7B model generates answer using both the query and retrieved documents
- System returns the response to the user


## **Results (System Performance)**

#### Faithfulness
Faithfulness measures how accurately the model’s responses reflect the provided input and context. In financial advisory, ensuring faithful responses is crucial to avoid misleading or incorrect guidance.

#### Baseline System Scores
- **Context Relevancy: 0.1203**
  - Indicates the extent to which the system’s answers incorporate relevant information from the prompt or context.
- **Context Recall: 0.1343**
  - Reflects how effectively the system retrieves or leverages any underlying knowledge base to inform its answers.
- **Faithfulness: 0.3365**
  - Assesses whether the generated answer is traceable and directly aligned with the query or context provided (lower suggests the baseline frequently introduces content not supported by the prompt).

These scores show that the baseline system, while functional, falls short in delivering consistent, context-driven recommendations. The relatively low Faithfulness score (0.3365) suggests answers may drift from the user’s specific prompt or lack sufficient supporting evidence.

## **Conclusion**

The baseline system demonstrates the feasibility of providing automated financial advice using a smaller language model, however, the low scores for Context Relevancy, Context Recall, and Faithfulness reveal clear limitations:
- Limited Contextual Depth: The system struggles to incorporate all relevant details from user queries or external data.
- Inconsistent Accuracy: Without advanced optimization or specialized reasoning frameworks, the model may produce incomplete or imprecise financial guidance.

These findings highlight the need for improved prompt engineering, data augmentation, and advanced reasoning methods to enhance system faithfulness and overall performance. This serves as the foundation upon which additional techniques—such as MIPROv2 optimization and Chain-of-Thought extraction—will be introduced in subsequent iterations.