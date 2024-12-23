import pandas as pd
import numpy as np
import os

# Globals
MAX_QUERY_LEN = 128
MAX_DOC_LEN = 4096


def get_docs(path):
    """Result is a dictionary with document_id as key and text as value.
       Example: {'1': 'text1', '2': 'text2', ...}
    """
    df = pd.read_csv(path)
    doc_ids = df['Codigo'].tolist()
    texts = df['text'].tolist()

    docs = {}
    for i in range(len(doc_ids)):
        docs[doc_ids[i]] = texts[i]

    return docs


def retrieve(queries, texts, doc_ids, embeddings_queries, embeddings_docs, top_k=10, sim_type='dot'):
    """
    Given a list of queries and documents, and their embeddings, compute the similarity between queries and documents
    and return the top_k most similar documents for each query.
    """
    import torch.nn.functional as F
    
    if sim_type == 'dot':
        similarity = embeddings_queries @ embeddings_docs.T
    elif sim_type == 'cosine':
        # Only use angle between embeddings
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)
        embeddings_docs = F.normalize(embeddings_docs, p=2, dim=1)
        similarity = (embeddings_queries @ embeddings_docs.T) * 100
    else:
        raise ValueError(f"Invalid similarity type: {sim_type}")
    # print("similarity", similarity)   # [[0.6265, 0.3477], [0.3499, 0.678 ]]
    
    similarity_dict = {}
    for i in range(len(queries)):
        # Sort similarity scores
        similarity_dict[queries[i]] = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)
        # Cut similarity scores to top_k results
        similarity_dict[queries[i]] = similarity_dict[queries[i]][:top_k]
        # Fill similarity_dict with the documents (for each query)
        # similarity_dict will be a dict with query as key and a list of
        # tuples of (similarity, document text, doc_id) as value
        for j, (doc_i, doc_sim) in enumerate(similarity_dict[queries[i]]):
            similarity_dict[queries[i]][j] += tuple([texts[doc_i], doc_ids[doc_i]])

    return similarity_dict


def embed_jinja(model, docs, queries, top_k=10):
    """
    Embed the queries and documents using the Jinja embeddings model and compute the similarity between queries and documents.
    """
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
    embeddings_queries = model.encode(queries, task="retrieval.query", max_length=MAX_QUERY_LEN)
    path = 'corpus/embeddings_corpus_jinja.npy'

    if not os.path.exists(path):
        texts = list(docs.values())
        embeddings_docs = model.encode(texts, task="retrieval.passage", max_length=MAX_DOC_LEN)
        # save embeddings
        np.save(path, embeddings_docs)
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    doc_ids = list(docs.keys())
    texts = list(docs.values())
    similarity_dict = retrieve(queries, texts, doc_ids, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict


def embed_bge(model, docs, queries, top_k=10):
    """
    Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
    """
    embeddings_queries = model.encode(queries, batch_size=2, max_length=MAX_QUERY_LEN)['dense_vecs']
    # Embed entire corpus if file does not exist
    path = 'corpus/embeddings_corpus_bge-m3.npy'
    if not os.path.exists(path):
        texts = list(docs.values())
        embeddings_docs = model.encode(texts, max_length=MAX_DOC_LEN)['dense_vecs']    # takes about 7min
        # save embeddings
        np.save(path, embeddings_docs)
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    doc_ids = list(docs.keys())
    texts = list(docs.values())
    similarity_dict = retrieve(queries, texts, doc_ids, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict


def embed_bm25(docs, queries, doc_ids, top_k=10):
    from rank_bm25 import BM25Okapi
    import numpy as np

    texts = list(docs.values())

    # Simple space-based tokenization
    tokenized_corpus = [text.lower().split() for text in texts]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Queries
    tokenized_queries = [query.lower().split() for query in queries]

    # key: query, value: (similarity, text, doc_id)
    similarity_dict = {}
    for tokenized_query, query in zip(tokenized_queries, queries):
        similarity_dict[query] = []
        
        # Calcular las puntuaciones BM25 para la consulta en cada documento
        scores = bm25.get_scores(tokenized_query)

        # Ordenar los documentos por relevancia
        sorted_docs = np.argsort(scores)[::-1]  # Orden descendente
        for idx in sorted_docs[:top_k]:
            value = (idx, scores[idx], texts[idx], doc_ids[idx])
            similarity_dict[query].append(value)
    return similarity_dict


def get_jinja_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return model


def get_bge_m3_model(checkpoint):
    from FlagEmbedding import BGEM3FlagModel

    # model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = BGEM3FlagModel(checkpoint) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


def main():
    from tqdm import tqdm
    import torch

    torch.cuda.empty_cache()

    # I want 30 annotated documents for each query
    top_k = 30

    docs = get_docs('corpus/corpus_google_min_line_len2_naive.csv')
    # assert len of texts is 5000
    assert len(docs) == 5000

    queries_path = 'corpus/queries_57.csv'
    queries = pd.read_csv(queries_path)["Query"].tolist()
    # assert len of queries is 57
    assert len(queries) == 57

    models = [
        "bm25",
        "bge-m3",
        "Jinja",
    ]

    device = torch.device("cuda")

    for model_name in tqdm(models):

        print(f"Retrieving documents using {model_name} model...")
        
        if model_name == "bm25":
            doc_ids = list(docs.keys())
            similarity_dict = embed_bm25(docs, queries, doc_ids, top_k)
        elif model_name == "bge-m3":
            checkpoint = 'BAAI/bge-m3'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            similarity_dict = embed_bge(model, docs, queries, top_k)

        elif model_name == "Jinja":
            model = get_jinja_model()
            model.to(device)
            similarity_dict = embed_jinja(model, docs, queries, top_k)
        
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        if len(similarity_dict) == 0:
            raise ValueError("No documents retrieved.")
        # Prepare a list of rows
        rows = []
        for query, doc_item in similarity_dict.items():
            for _, doc_sim, doc_text, doc_id in doc_item:
                rows.append({"Query": query, "doc_sim": doc_sim, "Codigo": doc_id, "doc_text": doc_text})

        # Create DataFrame
        df = pd.DataFrame(rows)

        path = f'corpus/retrieved_docs_{model_name}.csv'
        df.to_csv(path, index=False)

        print(f"Retrieved documents saved to {path}")
    
        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()