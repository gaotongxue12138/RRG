from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='描述你的脚本')
parser.add_argument('--dataset_name', type=str, help='帮助信息')
args = parser.parse_args()
name = args.dataset_name
df = pd.read_json('*', lines=True)
total_rows = len(df)
chunk_size = 50000
num_chunks = total_rows // chunk_size
if total_rows % chunk_size != 0:
    num_chunks += 1
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size
    chunk_df = df.iloc[start_idx:end_idx]
    output_file = f'*{i}.json'
    chunk_df.to_json(output_file, orient='records', lines=True)
embeddings = HuggingFaceEmbeddings(model_name = '*')
for i in range(0, num_chunks):
    loader = JSONLoader(
        file_path=f'*/{i}.json', 
        jq_schema='.nl',
        text_content=False,
        json_lines=True)
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(f'*')
base_db = FAISS.load_local(f'*',embeddings)
for i in range(1, num_chunks):
    current_db = FAISS.load_local(f'*',embeddings)
    base_db.merge_from(current_db)
base_db.save_local(f'*')