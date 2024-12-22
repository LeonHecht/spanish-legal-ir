"""
This file serves the purpose to merge the retrieval results of
several retrieval models for annotation.
"""

import pandas as pd


def read_retrieved_docs(path):
    df = pd.read_csv(path)
    return df


def get_queries():
    queries_path = 'corpus/queries_57.csv'
    queries = pd.read_csv(queries_path)["Consultas"].tolist()
    # assert len of queries is 57
    assert len(queries) == 57
    return queries


def main():
    """
    Input: Several different csv files with retrieved top k docs from different models
           for n queries.
           df with columns: Query, doc_sim, Codigo, doc_text
        
    Output: A single csv file with top k retrieved docs for each query.
            df with columns: Query, Rank, Codigo, doc_text
    """
    
    # I want 30 docs for each query for annotation 
    top_k = 30

    # get queries list
    queries = get_queries()

    paths = [
        "corpus/retrieved_docs_bge-m3.csv",
        "corpus/retrieved_docs_bm25.csv",
        "corpus/retrieved_docs_Jinja.csv",
    ]

    in_dfs = []
    for path in paths:
        df = read_retrieved_docs(path)
        in_dfs.append(df)
    
    # for each query there will be one out_df with top_k entries (docs)
    out_dfs = []
    for query in queries:
        # Get sub-dfs for current query
        sub_in_dfs = []
        for df in in_dfs:
            sub_df = df[df["Query"] == query]
            # assert len of sub_df is top_k
            assert len(sub_df) == top_k
            sub_in_dfs.append(sub_df)

        # Perform intersection between the 3 dfs
        # Convert rows to sets of tuples
        # CAUTION: Only works for 3 dfs!
        # Extract the "Codigo" column as sets
        set1 = set(sub_in_dfs[0]["Codigo"])
        set2 = set(sub_in_dfs[1]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set2 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        out_df = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)]
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]
        
        # Calculate docs/rows left to fill out_df up to top_k
        docs_left = top_k - len(out_df)
        if docs_left == 0:
            out_dfs.append(out_df)
            continue
            
        # Perform intersection between df 1 and 2
        set1 = set(sub_in_dfs[0]["Codigo"])
        set2 = set(sub_in_dfs[1]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set2

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]

        # Perform intersection between df 2 and 3
        set2 = set(sub_in_dfs[1]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set2 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[1][sub_in_dfs[1]["Codigo"].isin(common_codigos)]
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]

        # Perform intersection between df 1 and 3
        set1 = set(sub_in_dfs[0]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]

        docs_left = top_k - len(out_df)

        curr_index = 0
        while docs_left > 0:
            out_df.loc[len(out_df)] = sub_in_dfs[curr_index].iloc[0]
            # Remove the first row from sub_df
            sub_in_dfs[curr_index] = sub_in_dfs[curr_index].iloc[1:]
            docs_left -= 1

            # let curr_index oscillate between 0 and 2
            if curr_index < 2:
                curr_index += 1
            else:
                curr_index = 0

        out_dfs.append(out_df)

    # write out dfs to csv
    for out_df in out_dfs:
        assert len(out_df) == top_k
        query = out_df["Query"].iloc[0]
        path = f'corpus/merged_retrieved_docs_{query}.csv'
        out_df.to_csv(path, index=False)
        print(f"Merged retrieved documents saved to {path}")


if __name__ == '__main__':
    main()