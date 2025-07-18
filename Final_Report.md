Final System Report

Report describing the final system to be delivered, incorporating corrections and consistent details from the baseline.

1. Analytic Approach

Target Definition

The Final System builds upon the baseline goal of generating financial insights and recommendations, with a heightened focus on faithfulness. Specifically, it aims to provide accurate, context-rich advice by leveraging both a domain-fine-tuned Falcon 7B model and advanced prompt optimization techniques.

Inputs (Description)
	1.	User Query
	•	A text prompt submitted by investors or advisors, potentially vague or incomplete.
	•	In the final system, these queries are refined through an optimized Chain-of-Thought (CoT) before inference.
	2.	Knowledge Base (RAG + Qdrant Vector Database)
	•	The same RAG-based knowledge repository used in the baseline.
	•	Relevant documents—collected from various sources (e.g., Alpaca) and stored in the Qdrant vector database—are fetched based on query relevance.

What Kind of System Was Built?

A prompt-optimized advisory system that:
	•	Retains Falcon 7B as the underlying language model (with domain-specific fine-tuning, consistent with the baseline).
	•	Incorporates MIPROv2 to optimize prompts, improving faithfulness and contextual alignment.
	•	Maintains the baseline’s RAG workflow for retrieving relevant information from the Qdrant database.

2. Solution Description

Simple Solution Architecture
	1.	Data Sources
	•	User Queries from various investor profiles.
	•	Qdrant Vector DB storing relevant financial documents.
	2.	Solution Components
	1.	Prompt Optimizer (MIPROv2)
	•	Refines user queries and extracted context into a structured Chain-of-Thought.
	•	Tested in multiple configurations (Light, Medium, Heavy) for best faithfulness.
	2.	Falcon 7B Model
	•	Domain-fine-tuned, as in the baseline.
	•	Receives the optimized prompt to generate final answers.
	3.	RAG Layer
	•	Same retrieval process as baseline, where relevant documents from Qdrant are appended to the prompt.

[User Query] 
   |
   | ---> [RAG / Qdrant Vector DB] ---> [Relevant Docs]
   |
   v
[MIPROv2 Prompt Optimizer] 
   |
   v
[Falcon 7B Model (Inference)]
   |
   v
[Final Answer]

What Is the Output?

A context-enriched response that:
	•	Incorporates relevant details from the Qdrant vector DB.
	•	Demonstrates improved faithfulness to user queries, thanks to structured CoT reasoning.

3. Data

Source
	1.	Baseline Financial Q&A Dataset
	•	Same domain-specific questions used to fine-tune and evaluate Falcon 7B in the baseline.
	2.	Additional Synthetic Examples
	•	Generated (when necessary) to bolster coverage of edge cases or rarer financial scenarios.

Data Schema
	•	about_me: User’s scenario or risk profile.
	•	context: Relevant market or policy information retrieved via RAG.
	•	question: The user’s financial question or goal.
	•	answer (if labeled): Used for evaluation during optimization.

Sampling

At least 200+ diverse Q&A samples to ensure coverage across various financial sectors (stocks, ETFs, retirement, tax strategies, etc.). This helps MIPROv2 learn robust CoT patterns without overfitting to a narrow domain.

4. Algorithm

Description of Data Flow
	1.	Query + Document Retrieval (RAG)
	•	The user’s prompt triggers a search in the Qdrant vector database.
	•	The most relevant documents are appended as context.
	2.	Prompt Optimization (MIPROv2)
	•	Takes in the user query + retrieved context.
	•	Refines them into a structured Chain-of-Thought prompt.
	•	Runs through multiple configurations (Light, Medium, Heavy), measuring faithfulness to pick the best approach.
	3.	Inference with Falcon 7B
	•	The final optimized prompt is passed to the domain-fine-tuned Falcon 7B model.
	•	The model produces an answer that is more aligned with the user’s query and the supporting context.
	4.	Output
	•	The user receives an improved, faithful response that demonstrates stronger adherence to the original question and integrated knowledge.

5. Results

MIPROv2 Configurations & Faithfulness

Light Configuration
	•	Baseline: 68.8%
	•	Avg Minibatch Score: 66.66%
	•	Optimized: 71.6%

Medium Configuration
	•	Baseline: 68.8%
	•	Avg Minibatch Score: 74.96%
	•	Optimized: 75.18%

Heavy Configuration
	•	Baseline: 68.4%
	•	Avg Minibatch Score: 75.18%
	•	Optimized: 76.08% (highest overall)

	Observation: Across all configurations, the optimized approach outperforms baseline faithfulness scores.

Data Enrichment Example
	•	FROM: Minimal JSON with brief descriptions.
	•	TO: Enriched JSON containing deeper context and extended reasoning, helping the model remain closely aligned with user prompts.

Comparison of Key Metrics

Compared to the baseline system, the final system demonstrates notable improvements:

Metric	Baseline Score	Final System Score	Improvement
Context Relevancy	0.1203	0.1239	+3.0% (approx)
Context Recall	0.1343	0.1543	+14.9% (approx)
Faithfulness	0.3365	0.484	+43.8% (approx)

	•	Context Relevancy: Improved alignment of answers to user queries and relevant documents.
	•	Context Recall: More comprehensive use of the Qdrant-based data.
	•	Faithfulness: Substantial gains, ensuring outputs remain accurate and supported by the provided prompts.

Final Note

By integrating MIPROv2 prompt optimization and structured Chain-of-Thought reasoning on top of the existing baseline (domain-fine-tuned Falcon 7B + RAG from Qdrant), the final system addresses the key limitations of vague queries and inconsistent accuracy. The result is a scalable, trustworthy, and higher-performing solution that more effectively aligns with the user’s financial needs and context.