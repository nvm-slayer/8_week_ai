from langchain_text_splitters import (CharacterTextSplitter, RecursiveCharacterTextSplitter)

text="""Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation, commonly known as RAG, is a technique that combines information retrieval with large language models. Instead of relying entirely on the knowledge stored in a model's parameters, RAG allows the model to retrieve relevant information from an external knowledge source before generating an answer. This approach is particularly useful when working with private documents, frequently updated information, or domain-specific knowledge that the model might not have encountered during training.

A typical RAG pipeline consists of several stages: document ingestion, text preprocessing, chunking, embedding generation, vector storage, retrieval, and response generation. Each stage plays an important role in the overall quality of the system. Poor decisions during preprocessing or chunking can negatively affect retrieval accuracy, even when a powerful language model is used for generation.
Why Chunking Matters
Large documents are difficult to process as a single unit. Embedding models have input limitations, and retrieving an entire document when only a small section is relevant introduces unnecessary noise. Chunking solves this problem by dividing documents into smaller pieces that can be independently embedded and retrieved.

However, chunking is not simply about splitting text into equal-sized pieces. The goal is to preserve meaningful information while creating chunks that are useful for semantic retrieval. If a chunk is too small, it may lose important context. For example, a chunk containing only the phrase "it improves retrieval accuracy" does not explain what "it" refers to. On the other hand, excessively large chunks may contain multiple unrelated topics, making similarity search less precise.
Fixed-Size Chunking
Fixed-size chunking divides text into segments based on a predefined character count or token count. For example, a document might be divided into chunks of 500 characters with an overlap of 50 characters.

Overlap is useful because information near the end of one chunk may be related to information at the beginning of the next chunk. Without overlap, important context can be lost at chunk boundaries.

Consider the following example. A paragraph explains that embeddings convert text into numerical vectors and that these vectors represent semantic meaning. If a fixed-size boundary separates the explanation from the sentence describing semantic meaning, retrieving either chunk independently might provide incomplete information.

Fixed-size chunking is simple, predictable, and computationally inexpensive. However, it does not understand the structure of a document. It might split a sentence in half, separate a heading from its explanation, or divide a paragraph at an inappropriate location.
Recursive Character Text Splitting
Recursive character text splitting attempts to preserve the natural structure of text while respecting a maximum chunk size. Instead of immediately cutting text at an arbitrary character position, it uses a hierarchy of separators.

A common separator hierarchy is:
1. Double newline, which represents paragraph boundaries.
2. Single newline, which represents line boundaries.
3. Spaces, which help preserve word boundaries.
4. Individual characters, which serve as the final fallback.
The splitter first tries to divide the document using the largest structural separator. If the resulting sections are still too large, it recursively attempts smaller separators until the chunks satisfy the configured size constraint.

For example, suppose a document contains three paragraphs. If each paragraph fits within the maximum chunk size, the splitter can preserve those paragraphs as individual chunks. If one paragraph is too large, it can be divided into smaller sentences or sections without unnecessarily disturbing the other paragraphs.

This approach generally produces more coherent chunks than arbitrary fixed-size splitting. However, it does not guarantee perfect semantic boundaries. It only uses structural separators and does not inherently understand the meaning of the text.
Choosing the Right Chunking Strategy
There is no universally optimal chunk size or chunking strategy. The best configuration depends on the document type, embedding model, retrieval method, and expected user queries.

For technical documentation, preserving headings and paragraphs can be particularly valuable because explanations often depend on surrounding context. For transcripts, however, splitting based on timestamps or speaker turns might be more appropriate.

Chunking strategies should be evaluated using actual retrieval performance rather than assumptions. Useful evaluation criteria include retrieval precision, recall, context completeness, and answer quality.

A practical experiment involves processing the same document using multiple chunking strategies, generating embeddings for each configuration, and testing them with an identical set of questions. Comparing the retrieved chunks helps identify which strategy provides the most relevant context.

Ultimately, a good RAG system is not defined by its choice of vector database or language model alone. The quality of document processing and retrieval plays an equally important role in producing accurate and contextually relevant answers."""


# fixed_splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size=500,
#     chunk_overlap=0)

# for i, chunk in enumerate(fixed_splitter.split_text(text),1):
#     print(f"Chunk {i}:\n{chunk}")

# print("fixed-size chunking complete.\n")
# print("============================================================\n")

line_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50)

for i, chunk in enumerate(line_splitter.split_text(text),1):
    print(f"Chunk {i}:\n{chunk}")

print("line-based chunking complete.\n")
print("============================================================\n")

paragraph_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50)

for i, chunk in enumerate(paragraph_splitter.split_text(text),1):
    print(f"Chunk {i}:\n{chunk}")

print("paragraph-based chunking complete.\n")
print("============================================================\n")

# recursive_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50)
# for i,chunk in enumerate(recursive_splitter.split_text(text),1):
#     print(f"Chunk {i}:\n{chunk}")
# print("recursive chunking complete.\n")