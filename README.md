# AIM-Midterm

## Background and Context

The CEO and corporate, with permission of the board, have assembled a crack data science and engineering team to take advantage of RAG, agents, and all of the latest open-source technologies emerging in the industry.  This time it's for real though.  This time, the company is aiming squarely at some Return On Investment - some ROI - on its research and development dollars.

## The Problem

**You are an AI Solutions Engineer**.  You've worked directly with internal stakeholders to identify a problem: `people are concerned about the implications of AI, and no one seems to understand the right way to think about building ethical and useful AI applications for enterprises.` 

This is a big problem and one that is rapidly changing.  Several people you interviewed said that *they could benefit from a chatbot that helped them understand how the AI industry is evolving, especially as it relates to politics.*  Many are interested due to the current election cycle, but others feel that some of the best guidance is likely to come from the government.

## Task 1: Dealing with the Data

You identify the following important documents that, if used for context, you believe will help people understand what’s happening now:

1. 2022: [Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) (PDF)
2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (PDF)

Your boss, the SVP of Technology, green-lighted this project to drive the adoption of AI throughout the enterprise.  It will be a nice showpiece for the upcoming conference and the big AI initiative announcement the CEO is planning.

<aside>
📝

Task 1: Review the two PDFs and decide how best to chunk up the data with a single strategy to optimally answer the variety of questions you expect to receive from people.

*Hint: Create a list of potential questions that people are likely to ask!*

</aside>

✅ Deliverables:

**Describe the default chunking strategy that you will use.**

As a default, I decided to make use of the [RecursiveTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) from LangChain. LangChain describes this splitter as “the recommended one for generic text.” The PDFs we are working with are well structured, and seem like a good fit for this text splitter. For the purposes of this application I will make use of the default delimiters, but I’ll set the baseline `chunk_size=1000` and the `chunk_overlap=200`. 

**Articulate a chunking strategy that you would also like to test out.**

In addition to the default chunking strategy with a chunk size of 1000 and overlap of 200, I'd also like to test an alternative approach. This strategy increases the individual chunk size to 2000 while reducing the overlap to 100. 

**Describe how and why you made these decisions**

Why I chose the default

I chose the default chunking strategy to break our original documents into reasonably sized chunks while maintaining contextually relevant information together. Chunks that are too small risk separating key information, while overly large chunks could potentially overwhelm our model with excess information. Using our outlined chunking strategy, we split our documents into 524 chunks, striking a balance between information density and model manageability.

Why I chose the alternative

I aimed to examine the effects of increasing the amount of information retrieved and passed to the model during inference. From past experiences, I've observed that increasing the chunk size can improve the accuracy of the retrieved context. This, in turn, enhances the likelihood of generating correct output, as the model has access to more comprehensive context.

## Task 2: Building a Quick End-to-End Prototype

**You are an AI Systems Engineer**.  The SVP of Technology has tasked you with spinning up a quick RAG prototype for answering questions that internal stakeholders have about AI, using the data provided in Task 1.

<aside>
📝

Task 2: Build an end-to-end RAG application using an industry-standard open-source stack and your choice of commercial off-the-shelf models

</aside>

✅ Deliverables:

**Build a live public prototype on Hugging Face, and include the public URL link to your space.**

[Loom of E2E Prototype](https://www.loom.com/share/ea600c7b18a24314a330545382447dc1?sid=e6537647-d8ff-4c92-83cd-41e3ca14f105)

**How did you choose your stack, and why did you select each tool the way you did?**

The tools I used were LangChain for the app orchestration, Qdrant for the vectordb, OpenAI for the embedding and chat model, Chainlit for the UI, and HuggingFace to host the app.

LangChain

LangChain was chosen for the app orchestration because it simplifies the integration of different components in the RAG pipeline. Langchain provides a wide range of document loaders, text-splitting strategies, retrieval methods and integration with a variety of model and vector store providers. LangChain makes it easy to connect the retriever, the vector store, and the LLM, to managing the conversational flow. LangChain's modular design allows for flexibility in how queries are processed and results are generated, making it ideal for a complex, multi-step task like retrieval-augmented generation. LangChain also provides built-in methods to handle conversation and state memory through methods such as the `ConversationBufferMemory`.

Qdrant

Qdrant was chosen for the vector store due to its use of HNSW (Hierarchical Navigable Small World) as the default indexing method. This approach is highly efficient for approximate nearest neighbor (ANN) searches. Qdrant also provides up to 1GB of a free cluster, along with real-time updates to vectors and payloads, which can be particularly useful for should we need to embed and store new data that is being generated (see future considerations section for why this could be important).

OpenAI

OpenAI was selected for both the embedding and chat models. Their API offers scalability, facilitating easy adaptation as our application expands. Additionally, OpenAI consistently updates its models, ensuring we can upgrade our application's core components when more advanced or cost-effective options become available.

Chainlit

We chose Chainlit since it provides a built in chat interface that is already similar to known experiences - ChatGPT/Claude and widget Chat-bots found on most websites nowadays. Chainlit provides a familiar experience, which enhances user comfort and engagement with the system. Additionally, Chainlit provides seamless integration with LLM observability systems such as LiteralAI which can provide insights into the performance, user satisfaction, and cost of our application.

HuggingFace

We are hosting our application on HuggingFace due to flexibility HuggingFace offers, the free cloud storage, and version control that is necessary for testing and deploying applications.

## Task 3: Creating a Golden Test Data Set

**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set."

<aside>
📝

Task 3: Generate a synthetic test data set and baseline an initial evaluation

</aside>

✅ Deliverables:

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, answer relevancy, context precision, and context recall.  Provide a table of your output results.

| **Metric** | **Baseline** |
| --- | --- |
| faithfulness | 0.698407 |
| answer_relevancy | 0.946766 |
| context_recall | 0.855903 |
| context_precision | 0.903935 |
| answer_correctness | 0.648744 |
1. What conclusions can you draw about performance and effectiveness of your pipeline with this information?

Our baseline RAG pipeline excels at maintaining answer relevance and performs well in retrieving appropriate context, with both context recall and precision exceeding 0.85. However, our two primary areas for improvement lie in enhancing the faithfulness and correctness of our answers.

This means our baseline pipeline does well at generating answers relevant to the users query, retrieving relevant chunks to the question, and retrieving chunks that could contain the correct answer to the question. Hopefully our fine-tuned embedding model and our alternate chunking strategy can improve our RAG pipeline’s ability to produce correct and context faithful answers. 

## Task 4: Fine-Tuning Open-Source Embeddings

**You are an Machine Learning Engineer.**  The AI Evaluation and Performance Engineer has asked for your help in fine-tuning the embedding model used in their recent RAG application build.

<aside>
📝

Task 4: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

</aside>

✅ Deliverables

1. Swap out your existing embedding model for the new fine-tuned version.  Provide a link to your fine-tuned embedding model on the Hugging Face Hub.

[Fine-tuned embedding model](https://huggingface.co/XicoC/midterm-finetuned-arctic)

Description: This is a [**sentence-transformers**](https://www.sbert.net/) model finetuned from [**Snowflake/snowflake-arctic-embed-m**](https://huggingface.co/Snowflake/snowflake-arctic-embed-m). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

1. How did you choose the embedding model for this application?

I chose Snowflake/snowflake-arctic-embed-m for several reasons. First, my past experiences with this model have shown significant performance improvements after fine-tuning. The fine-tuning process itself is quick and cost-effective, which is a major advantage. Moreover, this model was pre-trained on a diverse corpus, providing us with a robust foundation for further fine-tuning. Finally, given its size and dimensionality, the model's base performance is already strong, indicating that fine-tuning could potentially boost our RAG system's performance even further. This impression is further reinforced by the model's popularity, evidenced by its **73,326** downloads in the past month alone.

## Task 5: Assessing Performance

**You are the AI Evaluation & Performance Engineer**.  It's time to assess all options for this product.

<aside>
📝

Task 5: Assess the performance of 1) the fine-tuned model, and 2) the two proposed chunking strategies

</aside>

✅ Deliverables

**Test the two chunking strategies using the RAGAS frameworks to quantify any improvements. Provide results in a table.**

Baseline: `chunk_size = 1000, chunk_overlap = 200`

MediumChunk: `chunk_size = 2000, chunk_overlap = 100`

| **Metric** | **Baseline** | **MediumChunk** | **Baseline -> MediumChunk** |
| --- | --- | --- | --- |
| faithfulness | 0.698407 | 0.895359 | 0.196952 |
| answer_relevancy | 0.946766 | 0.955419 | 0.008653 |
| context_recall | 0.855903 | 0.934028 | 0.078125 |
| context_precision | 0.903935 | 0.937500 | 0.033565 |
| answer_correctness | 0.648744 | 0.629267 | -0.019477 |

In this comparison, we evaluated the baseline chunking strategy and the alternate strategy. Here we can see we get a slight regression in answer_correctness however major improvements in the other 4 metrics with a huge improvement in faithfulness - likely due to the additional context the answer can be evaluated against. Additionally, there are decent sized gains in retrieval performance as shown in the improvements to context_recall and context_precision. Answer_relevancy sees minor improvements, but improvements nonetheless.

Winner: MediumChunk: `chunk_size = 2000, chunk_overlap = 100` 

**Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.**

Reg Embedding vs Fine-tune embedding model

| **Metric** | **Baseline** | **Fine-Tune Embedding** | **Baseline -> Fine-Tune Embedding** |
| --- | --- | --- | --- |
| faithfulness | 0.895359 | 0.868351 | -0.027007 |
| answer_relevancy | 0.955419 | 0.955777 | 0.000358 |
| context_recall | 0.934028 | 0.944444 | 0.010417 |
| context_precision | 0.937500 | 0.953668 | 0.016168 |
| answer_correctness | 0.629267 | 0.603407 | -0.025861 |

Now that we have the chunking size to use, it’s time to evaluate our RAG pipeline’s performance using OpenAI’s `text-embedding-3-small model` and our fine-tuned `snowflake-arctic-embed-m` model using the MediumChunk chunking strategy. As we can see from the results we see minor improvements in answer_relevancy, context_recall, and context_precision, however we see some notable regression in faithfulness and answer_correctness. Since our scores for answer_relevancy, context_recall, and context_precision were already strong in the Baseline test, we were more concerned with the regression in the other two areas, primarily answer_correctness. Due to this regression using the fine-tuned model, we decided to maintain the baseline embedding model and focus on improving the answer_correctness  method through other means.

Winner: Baseline: `OpenAIEmbeddings(model="text-embedding-3-small")` 

**Other Evaluations**

Baseline: `chunk_size = 1000, chunk_overlap = 200`

MediumChunk: `chunk_size = 1000, chunk_overlap = 200`

LargeChunk: `chunk_size = 3000, chunk_overlap = 0`

| **Metric** | **Baseline** | **MediumChunk** | **LargeChunk** | **HigestValue** |
| --- | --- | --- | --- | --- |
| faithfulness | 0.698407 | 0.895359 | 0.796131 | 0.9 (MediumChunk) |
| answer_relevancy | 0.946766 | 0.955419 | 0.959296 | 0.96 (LargeChunk) |
| context_recall | 0.855903 | 0.934028 | 0.843750 | 0.93 (MediumChunk) |
| context_precision | 0.903935 | 0.937500 | 0.929398 | 0.94 (MediumChunk) |
| answer_correctness | 0.648744 | 0.629267 | 0.632580 | 0.65 (Baseline) |

Now that our two initial tests showed that the MediumChunk strategy and using OpenAI’s text-embedding-3-small model produced the most optimal results, we wanted to ensure that our ‘winning’ chunking strategy was still the best possible chunking strategy. So we wanted to run an additional test where we would increase the chunk size and in turn potentially increase the amount of information passed to the model at inference. To do this, the document chunk sizes were increased by 50%, to 3000 - we removed any overlap since the chunk sizes were already considerably large compared to our ‘winning’ strategy and our baseline. After testing we can see that answer_relevancy and answer_correctness improved just slightly over our ‘winning’ strategy but the regression in context_recall was too large to notice, and we thought having the relevant answer in the context was too big a factor to overlook. We decided that the best chunking strategy was still the MediumChunk strategy of `chunk_size = 2000, chunk_overlap = 100` .

Winner: MediumChunk: `chunk_size = 2000, chunk_overlap = 100` 

Retrieval comparison where Baseline = `chunk_size = 2000, chunk_overlap = 100` embedding = `OpenAIEmbeddings(model="text-embedding-3-small")`  and retriever = `vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})` .

MultiQuery and ContextalCompression used the same chunking strategy and embedding model as the Baseline.

| **Metric** | **Baseline** | **MultiQuery** | **ContextualCompression** | **HigestValue** |
| --- | --- | --- | --- | --- |
| faithfulness | 0.895359 | 0.896804 | 0.749092 | 0.9 (MultiQuery) |
| answer_relevancy | 0.955419 | 0.953211 | 0.913993 | 0.96 (Baseline) |
| context_recall | 0.934028 | 0.890625 | 0.725694 | 0.93 (Baseline) |
| context_precision | 0.937500 | 0.920732 | 0.905093 | 0.94 (Baseline) |
| answer_correctness | 0.629267 | 0.690058 | 0.570685 | 0.69 (MultiQuery) |

With our Baseline already producing high scores for faithfulness, answer relevancy, context recall, and context precision, we aimed to improve answer correctness, as delivering accurate information to users is paramount. Our goal was to maintain scores of ~0.90 or better across these four "anchor" metrics while enhancing our RAG pipeline's ability to produce correct outputs. The evaluations showed that the MultiQuery retrieval strategy achieved this goal. The four anchor metrics remained around ~0.90, and answer correctness saw a significant increase. Context recall did regress slightly but remained close to our ~0.90 target. In contrast, the ContextualCompression strategy fell short of our objectives. Each metric, including answer correctness, saw considerable regression, with two of our anchor metrics declining by 0.15 (faithfulness) and 0.21 (context recall). This evaluation narrowed our choice to either the baseline vector store retrieval strategy or the MultiQuery retrieval strategy. After careful consideration, we chose the MultiQuery strategy as the winner. It met our goal of maintaining high scores across our anchor metrics while substantially improving our answer correctness scores.

Winner: MultiQuery Retrieval

The AI Solutions Engineer asks you “Which one is the best to test with internal stakeholders next week, and why?”

After testing out multiple configurations of our RAG pipeline, we have decided that the best configuration is to use the MediumChunkSize of `chunk_size = 2000, chunk_overlap = 100`, OpenAI’s `text-embedding-3-small` model, and a MultiQuery retriever (see configuration below). The aim of these evaluations is to find the most optimal configuration to produce the best performing RAG pipeline. Optimal in this case is subjective, especially when considering the performance across 5 different metrics. We settled on this 'Best' configuration due to its strong performance—scoring 0.89 or higher—across faithfulness, answer relevancy, context recall, and context precision. While it didn't excel in every metric (the Baseline config slightly outperformed in answer relevancy and context precision, and notably in context recall), the significant improvements in answer correctness were too substantial to ignore.

Best config:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10},
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)
```

## Task 6: Managing Your Boss and User Expectations

**You are the SVP of Technology**.  Given the work done by your team so far, you're now sitting down with the AI Solutions Engineer.  You have tasked the solutions engineer to test out the new application with at least 50 different internal stakeholders over the next month.

What is the story that you will give to the CEO to tell the whole company at the launch next month?

The team has deployed a RAG-based chat application that can answer a wide range of questions, helping the organization understand how the AI industry is evolving, especially in relation to politics. It's powered by information from the Biden-Harris Administration's Blueprint for an AI Bill of Rights and the National Institute of Standards and Technology's report on Generative Artificial Intelligence Profile. The team has worked diligently to build an application that sheds light on complex AI topics, including the government's steps to ensure a safe yet ambitious approach to AI, and its perceived impact on the American people.

There appears to be important information not included in our build, for instance, the [270-day update](https://www.whitehouse.gov/briefing-room/statements-releases/2024/07/26/fact-sheet-biden-harris-administration-announces-new-ai-actions-and-receives-additional-major-voluntary-commitment-on-ai/) on the 2023 executive order on [Safe, Secure, and Trustworthy AI](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/).  How might you incorporate relevant white-house briefing information into future versions?

To keep our application current, we propose implementing an automated update process for V0.1. This will utilize Qdrant's cluster feature, an LLM-based classification function, and a batch scraping function. We'll create a function to scrape releases from the White House's briefing room ([whitehouse.com/briefing-room](http://whitehouse.com/briefing-room)) nightly. An LLM will then filter the content to identify AI-related documents. For relevant releases, we'll chunk the article using our existing strategy and embed the new content into our Qdrant database cluster.

After chunking, we'll efficiently add only the new embeddings to the existing vector database without re-embedding previous content. This approach allows us to incrementally update our knowledge base, ensuring new information is available alongside existing content. This strategy keeps our RAG application up-to-date with the latest AI policies and executive orders while maintaining performance and storage efficiency.
