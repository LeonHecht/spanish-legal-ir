import pandas as pd
import numpy as np

# Globals
MAX_QUERY_LEN = 128
MAX_DOC_LEN = 4096


def get_docs(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    return texts


def retrieve(queries, docs, embeddings_queries, embeddings_docs, top_k=10, sim_type='dot'):
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
    print("similarity", similarity)   # [[0.6265, 0.3477], [0.3499, 0.678 ]]

    similarity_dict = {}
    for i in range(len(queries)):
        # Sort similarity scores
        similarity_dict[queries[i]] = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)
        # Cut similarity scores to top_k results
        similarity_dict[queries[i]] = similarity_dict[queries[i]][:top_k]
        # Fill similarity_dict with the documents (for each query)
        # similarity_dict will be a dict with query as key and a list of the document texts as value
        for j, (doc_i, doc_sim) in enumerate(similarity_dict[queries[i]]):
            similarity_dict[queries[i]][j] += tuple([docs[doc_i]])

    return similarity_dict


def embed_e5(model, docs, queries, top_k=10, tokenizer=None):
    from torch import Tensor

    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    query_texts = ["query: " + query for query in queries]
    doc_texts = ["passage: " + doc for doc in docs]

    # Tokenize the input texts
    batch_dict_queries = tokenizer(query_texts, max_length=MAX_QUERY_LEN, padding=True, truncation=True, return_tensors='pt')
    batch_dict_docs = tokenizer(doc_texts, max_length=MAX_DOC_LEN, padding=True, truncation=True, return_tensors='pt')

    outputs_queries = model(**batch_dict_queries)
    embeddings_queries = average_pool(outputs_queries.last_hidden_state, batch_dict_queries['attention_mask'])
    
    outputs_docs = model(**batch_dict_docs)
    embeddings_docs = average_pool(outputs_docs.last_hidden_state, batch_dict_docs['attention_mask'])

    # Compute similarities
    similarity_dict = retrieve(queries, docs, embeddings_queries, embeddings_docs, top_k, sim_type='cosine')
    return similarity_dict


def embed_jinja(model, docs, queries, top_k=10):
    """
    Embed the queries and documents using the Jinja embeddings model and compute the similarity between queries and documents.
    """
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
    embeddings_queries = model.encode(queries, task="retrieval.query", max_length=MAX_QUERY_LEN)
    embeddings_docs = model.encode(docs, task="retrieval.passage", max_length=MAX_DOC_LEN)

    # Compute similarities
    similarity_dict = retrieve(queries, docs, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict



def embed_baai(model, docs, queries, top_k=10, model_name='bge-m3'):
    """
    Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
    """
    import os

    embeddings_queries = model.encode(queries, 
                                batch_size=12, 
                                max_length=MAX_QUERY_LEN,
                                )['dense_vecs']
    # Embed entire corpus if file does not exist
    if model_name == 'bge-m3':
        if not os.path.exists('embeddings_corpus_bge-m3.npy'):
            embeddings_docs = model.encode(docs, max_length=MAX_DOC_LEN)['dense_vecs']    # takes about 7min
            # save embeddings
            np.save('embeddings_corpus_gemma2.npy', embeddings_docs)
        else:
            # Load embeddings
            embeddings_docs = np.load('embeddings_corpus_bge-m3.npy')
    elif model_name == 'gemma2':
        if not os.path.exists('embeddings_corpus_gemma2.npy'):
            embeddings_docs = model.encode(docs, max_length=MAX_DOC_LEN)['dense_vecs']
            # save embeddings
            np.save('embeddings_corpus_gemma2.npy', embeddings_docs)
        else:
            # Load embeddings
            embeddings_docs = np.load('embeddings_corpus_gemma2.npy')
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # Compute similarities
    similarity_dict = retrieve(queries, docs, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict


def get_jinja_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return model


def get_e5_model():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    return model, tokenizer


def get_gemma2_model():
    from FlagEmbedding import FlagLLMModel

    model = FlagLLMModel(
        'BAAI/bge-multilingual-gemma2',
        # query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
        query_instruction_for_retrieval="Dado una consulta de búsqueda de documentos legales, recupera documentos relevantes que respondan a dicha consulta.",
        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    return model


def get_bge_m3_model(checkpoint):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


def main():
    from tqdm import tqdm

    texts = get_docs('corpus/corpus_google_min_line_len2_naive.csv')

    queries = [
        "Extradición",
        "Robo con arma de fuego",
    ]

    # Get models
    checkpoint = 'BAAI/bge-m3'
    # checkpoint = 'BAAI/bge-m3-unsupervised'
    # checkpoint = 'BAAI/bge-m3-retromae'
    model_bge = get_bge_m3_model(checkpoint)
    
    model_gemma2 = get_gemma2_model()
    model_e5, tokenizer_e5 = get_e5_model()
    model_jinja = get_jinja_model()

    models = {
        "gemma2": model_gemma2,
        "bge-m3": model_bge,
        "E5": model_e5,
        "Jinja": model_jinja,
    }

    for model_name, model in tqdm(models.items()):
        if model_name == "bge-m3":
            similarity_dict = embed_baai(model, texts, queries, top_k=10)
        elif model_name == "gemma2":
            similarity_dict = embed_baai(model, texts, queries, top_k=10)
        elif model_name == "E5":
            similarity_dict = embed_e5(model, texts, queries, top_k=10, tokenizer=tokenizer_e5)
        elif model_name == "Jinja":
            similarity_dict = embed_jinja(model, texts, queries, top_k=10)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        if len(similarity_dict) == 0:
            raise ValueError("No documents retrieved.")
        # Prepare a list of rows
        rows = []
        for query, docs in similarity_dict.items():
            for doc_id, doc_sim, doc_text in docs:
                rows.append({"Query": query, "doc_sim": doc_sim, "doc_text": doc_text})

        # Create DataFrame
        df = pd.DataFrame(rows)

        path = f'retrieved_docs_{model_name}.csv'
        df.to_csv(path, index=False)

        print(f"Retrieved documents saved to {path}")


if __name__ == '__main__':
    main()