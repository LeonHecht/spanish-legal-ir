import pandas as pd
import json


def read_corpus_csv(path):
    # corpus/corpus_google_min_line_len2_naive.csv
    # get all texts from the corpus
    df = pd.read_csv(path, usecols=['Codigo', 'text'])
    # rename Codigo column to document_id
    df = df.rename(columns={'Codigo': 'document_id'})
    # convert document_id to string
    df['document_id'] = df['document_id'].astype(str)
    return df


def get_queries_df():
    queries_path = 'corpus/queries_57.csv'
    queries = pd.read_csv(queries_path)
    # assert len of queries is 57
    assert len(queries) == 57
    return queries


def create_corpus_file(out_path):
    # corpus/docTAG/corpus_doctag_no_spanish_with_resuelve.json
    df = read_corpus_csv('corpus/corpus_google_min_line_len2_naive.csv')
    # Convert DataFrame to list of dictionaries
    collection_list = df.to_dict(orient='records')
    # Create the final JSON structure
    json_data = {"collection": collection_list}
    # Convert the structure to JSON string
    json_output = json.dumps(json_data, indent=2, ensure_ascii=False)

    # Save the result to a JSON file (optional)
    with open(out_path, 'w') as file:
        file.write(json_output)
    print(f"File created: {out_path}")
    return json_output


def create_query_file(queries_path, out_path):
    # "corpus/queries.csv"
    # "corpus/docTAG/topics_57.json"
    topics = []

    df = pd.read_csv(queries_path)
    
    for row in df.itertuples():
        topic_id, query = row[1], row[2]
        
        # Create a dictionary for the current topic
        topic = {
            "topic_id": topic_id,
            "title": query,
            "description": "",
            "narrative": ""
        }
        
        # Add the topic to the list
        topics.append(topic)
    
    # Create the final JSON structure
    data = {"topics": topics}
    
    # Write the data to a JSON file
    with open(out_path, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

    print(f"JSON file '{out_path}' has been created.")


def create_runs_file(out_path):
    """ What documents will be available for annotation for each query
        out_path should be 'corpus/docTAG/runs.json'
    """
    df_queries = get_queries_df()

    # Create an empty dictionary for the output structure
    output = {"run": []}

    # for topic_id, query in queries:
    for _, row in df_queries.iterrows():
        topic_id = row['topic_id']
        query = row['Query']

        path = f"corpus/retrieved_docs_for_annotation/merged_retrieved_docs_{query}.csv"
        df = pd.read_csv(path)

        # rename Codigo column to document_id
        df = df.rename(columns={'Codigo': 'document_id'})



        documents = []
        for _, df_row in df.iterrows():
            doc_id = df_row['document_id']
            documents.append({'document_id': doc_id})
        
        output['run'].append({
            "topic_id": topic_id,
            "documents": documents
        })
    
    # Convert the result to a JSON formatted string with indentation
    json_output = json.dumps(output, indent=2)

    # Save the result to a JSON file (optional)
    with open(out_path, 'w') as file:
        file.write(json_output)

    print(f"File created: {out_path}")


def create_labels_file(out_path):
    text = """
    {
    "labels": [
        "Altamente relevante",
        "Relevante",
        "Parcialmente relevante",
        "No relevante"
    ]
    }    
    """

    with open(out_path, 'w') as file:
        file.write(text)

    print(f"File created: {out_path}")


def main():
    create_corpus_file("corpus/docTAG/corpus_docTAG.json")
    create_query_file("corpus/queries_57.csv", "corpus/docTAG/topics_57.json")
    create_runs_file("corpus/docTAG/runs.json")
    create_labels_file("corpus/docTAG/labels.json")

if __name__ == "__main__":
    main()