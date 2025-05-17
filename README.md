# README  
  

## Preference-Guided Refactored Tuning for Retrieval Augmented Code Generation 
https://arxiv.org/pdf/2409.15895

## Project Overview  
  
This repository hosts the code for our project, which is primarily composed of two components: the **Retriever** and the **Generator**.  
  
## Retriever  
  
The retriever's functionality is divided into two main steps:  
  
1. **Vector Database Creation**:  
   - Utilize the `build-vec-database.py` script to establish an offline vector database that facilitates efficient vector retrieval.  
  
2. **Dual-Stage Retrieval Process**:  
   - Execute the `dual-stage-retriever.py` script to perform a two-stage retrieval.  
     - In the first stage, vector retrieval is employed to recall `K1` pertinent documents.  
     - The second stage involves sparse retrieval, further refining the recall based on the initial set of documents.  
  
## Generator  
  
The generator component encompasses two distinct phases:  
  
### Phase 1: Model Training  
  
- **Refactorer Model Training**:  
  - Run the `stage1-refactorer.py` script to train the code refactoring model, designated as `refactorer`.  
  
- **Generator Model Training**:  
  - Execute the `stage1-generator.py` script to train the code generation model, identified as `generator`.  
  
### Phase 2: Preference Optimization  
  
- Implement `stage2-preference.py` to refine the `refactorer` based on feedback provided by the `generator`.  
  - This phase aims to enhance the `refactorer`'s alignment with the `generator`'s preferences, thereby boosting the overall system performance.  
  
## Project File Structure  
  
```plaintext  
project-root/  
│  
├── README.md  
├── retriever/  
│   ├── build-vec-database.py  
│   └── dual-stage-retriever.py  
└── generator/  
    ├── stage1-refactorer.py  
    ├── stage1-generator.py  
    └── stage2-preference.py