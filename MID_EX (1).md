# **Project Charter**

## **Business Background**

### **The Client & Business Domain**

The client operates within the wealth management and investment advisory space—this could be a traditional brokerage, a fintech startup, or an asset management firm. In this domain, delivering timely and data-driven insights is crucial for guiding high-stakes financial decisions. However, many end users in these firms (from retail investors to mid-level advisors) may lack the depth of financial expertise to pose targeted questions, hindering the effectiveness of AI-driven advisory tools. 

Additionally, the firm faces cost and scalability constraints when considering large-scale model fine-tuning, often resorting to smaller or open-source models that may not match the performance of more robust, proprietary solutions. Balancing user-centric guidance with the technological limitations of AI infrastructure is therefore a key challenge for any enterprise operating in this highly competitive sector.

### **Business Problems & Key Performance Indicators**

1. **Communication Gap Leading to Vague Queries**
   - Problem: Many users lack the financial expertise to frame precise and well-defined questions. This results in ambiguous or incomplete inputs to AI-driven advisors, causing non-optimal or inaccurate recommendations.  
   - Impact: Inconsistent and often subpar guidance undermines user trust and can lead to poorly informed financial decisions.  
   - KPIs:  
     - Query Clarity Index: Percentage of user queries that must be revised or clarified before producing meaningful advice.  
     - Guidance Accuracy: Rate at which AI-generated recommendations align with expert or market-validated outcomes.

2. **High Cost/Feasibility Constraints of Large-Scale Model Fine-Tuning**
   - Problem: Fine-tuning large-scale LLMs (e.g., GPT-like models) on specialized financial data is prohibitively expensive for most companies, both in terms of computing resources and data-privacy compliance.  
   - Impact: Organizations either forego advanced personalization or rely on small-scale alternatives — limiting the depth and sophistication of AI insights.  
   - KPIs:  
     - Model Training Cost: Total expenditures on hardware, software, and personnel required for model development and updates.  
     - Time-to-Market for Updates: How quickly new financial insights or model improvements can be integrated and deployed.

3. **Performance Gaps in Smaller or Open-Source Models**
   - Problem: Although smaller or open-source models are more cost-effective to fine-tune and deploy, they often fall short in capturing context-rich financial insights compared to larger, state-of-the-art models.  
   - Impact: Suboptimal performance leads to lower client satisfaction, decreased adoption of AI solutions, and potential financial losses if the guidance is inaccurate.  
   - KPIs:  
     - Accuracy Score: Degree to which model recommendations mirror real-world market movements or expert consensus.  
     - Client Satisfaction / NPS (Net Promoter Score): User feedback metric indicating the perceived quality and trustworthiness of AI-delivered insights.

## **Scope**

Our approach addresses the cost and performance trade-offs of large model fine-tuning by integrating prompt optimization into the workflow:

1. **Leveraging Larger Models for Prompt Design**
   - Instead of performing expensive fine-tuning on massive LLMs, we extract reasoning structures (Chain-of-Thought, CoT) from a stronger model during a prompt optimization phase. This allows us to capture high-level logic and domain knowledge without incurring the cost of full-scale training.

2. **MIPROv2 for Faithfulness**
   - We apply the MIPROv2 algorithm to refine and optimize these Chain-of-Thought programs so that the smaller model’s responses are both contextually rich and faithfully aligned with the user’s prompt. This ensures the generated advice maintains accuracy and trustworthiness.

3. **Integration into Smaller Models**
   - These optimized prompts—containing proven reasoning paths—are then fed to smaller, more cost-effective LLMs. By emulating the CoT from the larger model, the smaller models can deliver significantly improved outputs without extensive training overhead.

Overall, the scope of our modifications is to bridge the gap between cost-efficiency and high-quality AI advisory services, focusing on *prompt engineering* rather than heavy fine-tuning.  

### **Customer Usage**

1. **Enhanced Query Interactions**
   - End-users continue to submit financial questions via chat-style interfaces. Now, behind the scenes, advanced prompt optimization techniques automatically structure and refine the user’s request, improving the clarity and accuracy of the smaller model’s output.

2. **Improved Advisory Workflows**
   - Any existing integrations—such as robo-advisory platforms or enterprise CRMs—gain from higher-fidelity insights because the smaller model can better replicate the reasoning of a larger LLM. This leads to more *faithful*, context-rich recommendations.

3. **Scalable, Cost-Effective Intelligence**
   - By removing the need to fine-tune large models directly, firms can maintain lower operational costs while delivering near state-of-the-art performance. This makes advanced AI advisory accessible to a broader range of financial services and user types.

Essentially, clients and end-users experience more accurate, trustworthy financial guidance without incurring the prohibitive costs typically associated with large-scale model tuning.

## Plan

### **1. Aligning Faithfulness to the Business Problem**

- Why Faithfulness Matters  
  Faithfulness—ensuring that every piece of AI-generated advice can be traced back to valid reasoning and the original prompt—directly addresses the inconsistent quality and context issues identified in our business problems. By increasing the accuracy and reliability of outputs, we reduce user confusion, build trust, and mitigate the risk of providing incomplete or misleading financial advice.  
  - *Example*: If users ask vague or incomplete questions, faithfully generated responses prevent speculation or irrelevant guidance, thereby improving client satisfaction and reducing risk.

### **2. Generating a Comprehensive Dataset (200+ Samples)**

- Data Collection Strategy  
  To support the MIPROv2 algorithm without risking overfitting, we compiled a diverse dataset of more than 200 validated financial Q&A samples. This dataset includes:
  1. **Intenal Dataset**: Internal dataset used by the company to finetune thier model.
  2. **Synthetic Dataset Generation**: Data samples generated by LLM based on fewshot from the internal dataset examples (randomly chosen). Made in order to collect enough samples in cases of small internal dataset.

- Why 200+ Matters  
  Having a large, varied set of queries and answers mitigates the risk of MIPROv2 focusing too narrowly on any single question type or domain segment. This ensures the optimized prompts perform robustly across a wide spectrum of financial scenarios.

### 3. **Utilizing a Stronger Model for Automatic Prompt Engineering**

- Role of the “Teacher” Model  
  We employ a larger, more capable LLM (e.g., GPT) to generate and refine Chain-of-Thought (CoT) prompts. This is crucial because:
  1. **Advanced Reasoning Structures**: Larger models demonstrate more sophisticated reasoning and context handling, providing a rich CoT template for smaller models to emulate.  
  2. **Reduced Trial and Error**: Relying on a stronger model’s reasoning cuts down on guesswork in crafting effective prompts—expediting the optimization cycle and boosting initial accuracy.
  
- How It Works  
  1. **Extraction**: GPT analyzes the financial dataset and user queries, identifying potential logical steps, clarifications, and domain-specific nuances.  
  2. **Refinement**: MIPROv2 then optimizes these extracted reasoning steps—focusing on faithfulness—so that prompts remain accurate and aligned with user inputs.  
  3. **Deployment**: The optimized CoT program is incorporated into smaller, cost-effective models (e.g., Falcon 7B) without extensive fine-tuning, thereby bridging performance gaps.

### **4. Integrating the Optimized Program into the Inference Pipeline**

1. **User Input + RAG Context**
   - When a user poses a financial question, relevant context is fetched from existing data sources (VectorDBs, knowledge bases) using Retrieval-Augmented Generation (RAG).  
2. **Optimized Prompt Execution**
   - The system uses the MIPROv2-enhanced Chain-of-Thought to transform the user’s question and context into a highly targeted prompt.  
3. **Smaller Model Inference**
   - This optimized prompt is then passed to the smaller, fine-tuned model, which emulates the refined reasoning to produce faithful, context-rich answers.  
4. **User Response**
   - The user receives an accurate and transparent recommendation, with minimal guesswork and maximal alignment to their original query.

This plan not only mitigates inconsistent quality of financial outputs but also reduces the operational overhead of training and deployment—ultimately benefiting both end-users and the firm’s bottom line.

### **Step-by-Step Process**

1. **Initial Setup (Pre deployment)**
   - Collect a 200+ Q&A dataset plus representative user queries.    
   - Build and Optimize Chain-of-Thought (CoT) program using MIPROv2 for higher faithfulness and context alignment.

2. **Inference**
   - Load optimized Chain-of-Thought (CoT) program.
   - Input user query and context to Chain-of-Thought (CoT) program to recieve the optimized prompt.  
   - Forward the optimized prompt to the Falcon model.

### **Future Plan**

1. **Generalize Target Market**
   We plan to expand our solution beyond the financial industry by leveraging its domain-agnostic design. This universal approach amplifies its market appeal, enabling widespread adoption without needing specialized data. Consequently, our solution can cater to varied sectors—from healthcare to retail—offering consistent performance improvements across different LLM use cases.

2. **Privacy-Preserving & Model-Agnostic Optimization**
   Our focus is to deliver a privacy-centric, in-house framework compatible with any AI model architecture. This ensures data remains strictly under the client’s control while still benefitting from our prompt optimization capabilities. As a result, organizations with stringent governance requirements can seamlessly integrate our solution, maximizing efficiency without compromising on security or compliance.

3. **Online Optimizer Update**
   We aim to develop a real-time updating mechanism for the optimizer, allowing it to evolve continuously based on dynamic user interactions and feedback. This incremental learning capability aligns the system with emerging trends and business objectives, maintaining robust, up-to-date performance without the need for prolonged retraining cycles.

## **Metrics**

### **Qualitative Objectives**
Increase user trust in system's financial advices and reduce factual inaccuracies (hallucinations).

### **Quantifiable Metric**
Faithfulness Score: Ratio of verifiable, context-supported claims to total claims in LLM answers.

### **Measurement Method**
Use RAGAS or similar evaluation tools to assess faithfulness and hallucination rates before and after prompt optimization.

## **Architecture**

Our architecture centers on two main components—a Prompt Optimizer and an Inference Pipeline—designed to continuously refine smaller, fine-tuned AI models and deliver high-quality financial advice.

### **Build Optimizer**
We begin by generating a dataset of at least 200 validated samples representing diverse financial queries and scenarios. This dataset is then used alongside GPT models to construct and optimize a Chain of Thought (CoT) program. The MIPROv2 algorithm, with faithfulness as its key metric, refines these reasoning steps to ensure the model’s outputs accurately align with user prompts. Once optimized, the CoT program is stored and made available for seamless integration in production workflows, eliminating the need for repeated large-scale model fine-tuning.

![Build Optimizer Architecture](image.png)

### **Inference Pipeline**  
During user interactions, relevant contextual data is retrieved through RAG (Retrieval-Augmented Generation) and appended to the prompt. The optimized CoT program is then executed, leveraging GPT’s advanced reasoning to refine the user’s query before passing it to the company’s smaller, fine-tuned AI model. The AI system subsequently produces tailored, clear, and valuable responses. By incorporating live user feedback into future CoT improvements, the pipeline maintains continuous alignment with evolving market conditions and user needs.
In combining prompt optimization (via GPT and MIPROv2) with a streamlined inference phase, this architecture ensures consistently faithful, context-rich recommendations without incurring the high costs of large-scale model tuning.


![Inference Pipeline Architecture](image-1.png)


## **Team Members**
- Limor Hodory
- Liav Katry
- Niv Vaknin
- Lior Abuhav
  
**Team Communication**
- Weekly Teams meetings for coordination and updates.

## **Customer Communication Strategy**
- Customers can contact us via the suppoet email at Support@FinovistaAI.com
- Customers can open a support ticket via our support platform at FinovistaAi.com

