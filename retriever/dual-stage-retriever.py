import pandas as pd
from rank_bm25 import BM25Okapi
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
import pandas as pd
import swifter
import argparse
parser = argparse.ArgumentParser(description='描述你的脚本')
parser.add_argument('--dataset_name', type=str, help='帮助信息')
args = parser.parse_args()
name = args.dataset_name
embeddings = HuggingFaceEmbeddings(model_name = '*')
json_dir = f'*'
df = pd.read_json(f'*', lines=True)
total_rows = len(df)
chunk_size = 50000
num_chunks = total_rows // chunk_size
if total_rows % chunk_size != 0:
    num_chunks += 1
l = num_chunks
file_range = range(0, l) 
json_dataframes = {}
for i in file_range:
    file_path = f'{json_dir}/{i}.json'
    df = pd.read_json(file_path, lines=True)
    json_dataframes[f'{i}.json'] = df
db = FAISS.load_local('*', embeddings)
retriever = db.as_retriever(search_kwargs={"k": 11})
def v_search(querys,mode):
    re_code = []
    re_nl = []
    for i in tqdm(range(len(querys)), desc="Processing"):
        docs = retriever.get_relevant_documents(querys['nl'][i])
        relevant_code = ""
        relevant_nl = ""
        for doc in docs:
            metadata = doc.metadata
            source_filename = metadata['source'].split('/')[-1]
            seq_num = metadata['seq_num']
            if source_filename in json_dataframes:
                relevant_code += json_dataframes[source_filename]['code'][seq_num - 1] + "<pad>"
                relevant_nl += json_dataframes[source_filename]['nl'][seq_num - 1] + "<pad>"
            else:
                print(f"File {source_filename} not found in preloaded data.")
        re_nl.append(relevant_nl)
        re_code.append(relevant_code)
    querys['relevant-10nl'] = re_nl
    querys['relevant-10code'] = re_code
    querys.to_json(f'*', orient='records',lines = True)
querys_train = pd.read_json(f"*",lines = True)
querys_test = pd.read_json(f"*", lines=True)
v_search(querys_train,'train')
v_search(querys_test,'test')
def get_bm25(df):
    corpus = df['relevant-10nl'].split('?_split_?')[:-1]
    codes = df['relevant-10code'].split('?_split_?')[:-1]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = df['nl'].split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_3_values = sorted([(score, index) for index, score in enumerate(doc_scores)], reverse=True)[1:4]
    top_3_indices = [index for score, index in top_3_values]
    ans = " ".join([codes[i] + "?_split_?" for i in top_3_indices])
    return ans
querys_train = pd.read_json(f'*',lines = True)
querys_test  = pd.read_json(f'*',lines = True)
querys_train['relevant'] = querys_train.swifter.apply(get_bm25,axis=1)
querys_test['relevant'] = querys_test.swifter.apply(get_bm25,axis=1)
querys_train.to_json(f'*', orient='records', lines=True)
querys_test.to_json(f'*', orient='records', lines=True)