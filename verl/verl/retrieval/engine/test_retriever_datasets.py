
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from retriever import FaissRetriever

def test_msmarco():
    print("\n" + "="*80)
    print("Testing MSMARCO Dataset")
    print("="*80)
    try:
        retriever = FaissRetriever(dataset="msmarco", verbose=True, index_device="cpu")
        print("Successfully initialized MSMARCO retriever")
        
        query = ["what is deep learning"]
        results = retriever.search(query, k=3)
        print(f"Query: {query}")
        ids = retriever.map_indices_to_ids(results[1])
        print(f"Top 3 IDs: {ids}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_fiqua():
    print("\n" + "="*80)
    print("Testing FiQA Dataset")
    print("="*80)
    try:
        retriever = FaissRetriever(dataset="fiqua", verbose=True, index_device="cpu")
        print("Successfully initialized FiQA retriever")
        
        query = ["what is net income"]
        results = retriever.search(query, k=3)
        print(f"Query: {query}")
        ids = retriever.map_indices_to_ids(results[1])
        print(f"Top 3 IDs: {ids}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test FiQA first (smaller)
    test_fiqua()
    
    # Test MSMARCO
    test_msmarco()
