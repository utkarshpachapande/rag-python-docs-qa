import os
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def load_documentation_data(save_path="data/raw_docs.json"):
    """
    Scrapes key documentation pages and assigns metadata for the dashboard.
    Iterates through all returned documents to ensure no library content is missed.
    """
    
    # Starter URLs mapped to their respective library names and types
    url_map = {
    # Add 2-3 specific pages for the "small" libraries
    "https://pandas.pydata.org/docs/user_guide/10min.html": {"library": "pandas", "type": "guide"},
    "https://numpy.org/doc/stable/user/absolute_beginners.html": {"library": "numpy", "type": "guide"},
    "https://scikit-learn.org/stable/tutorial/basic/tutorial.html": {"library": "sklearn", "type": "tutorial"},
    "https://scikit-learn.org/stable/modules/clustering.html": {"library": "sklearn", "type": "guide"},
    "https://matplotlib.org/stable/tutorials/introductory/quick_start.html": {"library": "matplotlib", "type": "guide"},
    "https://matplotlib.org/stable/tutorials/introductory/sample_plots.html": {"library": "matplotlib", "type": "guide"},
    "https://matplotlib.org/stable/users/explain/base/index.html": {"library": "matplotlib", "type": "guide"},
    "https://matplotlib.org/stable/tutorials/introductory/pyplot.html": {"library": "matplotlib", "type": "guide"},
    "https://matplotlib.org/stable/users/explain/axes/index.html": {"library": "matplotlib", "type": "guide"},

   # --- GENERAL PYTHON BASICS (To answer syntax & "Hello World" questions) ---
        "https://docs.python.org/3/tutorial/introduction.html": {"library": "python_core", "type": "tutorial"},
        "https://docs.python.org/3/tutorial/controlflow.html": {"library": "python_core", "type": "tutorial"},
        "https://docs.python.org/3/tutorial/datastructures.html": {"library": "python_core", "type": "tutorial"},
        "https://docs.python.org/3/library/functions.html": {"library": "python_core", "type": "reference"},

        # --- PANDAS (Expanded) ---
        "https://pandas.pydata.org/docs/user_guide/10min.html": {"library": "pandas", "type": "guide"},
        "https://pandas.pydata.org/docs/user_guide/indexing.html": {"library": "pandas", "type": "guide"},
        "https://pandas.pydata.org/docs/user_guide/missing_data.html": {"library": "pandas", "type": "guide"},

        # --- NUMPY (Expanded) ---
        "https://numpy.org/doc/stable/user/absolute_beginners.html": {"library": "numpy", "type": "guide"},
        "https://numpy.org/doc/stable/user/basics.creation.html": {"library": "numpy", "type": "guide"},
        "https://numpy.org/doc/stable/user/basics.indexing.html": {"library": "numpy", "type": "guide"},

        # --- SCIKIT-LEARN (Expanded) ---
        "https://scikit-learn.org/stable/tutorial/basic/tutorial.html": {"library": "sklearn", "type": "tutorial"},
        "https://scikit-learn.org/stable/modules/clustering.html": {"library": "sklearn", "type": "guide"},
        "https://scikit-learn.org/stable/modules/linear_model.html": {"library": "sklearn", "type": "guide"},
        "https://scikit-learn.org/stable/modules/ensemble.html": {"library": "sklearn", "type": "guide"},

        # --- MATPLOTLIB (Expanded) ---
        "https://matplotlib.org/stable/tutorials/introductory/quick_start.html": {"library": "matplotlib", "type": "guide"},
        "https://matplotlib.org/stable/tutorials/introductory/pyplot.html": {"library": "matplotlib", "type": "guide"},
        "https://matplotlib.org/stable/users/explain/axes/axes_intro.html": {"library": "matplotlib", "type": "guide"} 
}
    
    refined_docs = []
    
    print("🔍 Starting targeted scraping...")
    
    for url, info in url_map.items():
        try:
            print(f"📡 Scraping {info['library']}...")
            
            # Initialize loader for the specific URL
            loader = WebBaseLoader(url)
            # Set a small delay to be polite to the documentation servers
            loader.requests_per_second = 1 
            
            # Load all content from the URL
            raw_doc_list = loader.load()
            
            if raw_doc_list:
                for doc in raw_doc_list:
                    # Only process if there is actual text content
                    if doc.page_content.strip():
                        new_doc = Document(
                            page_content=doc.page_content,
                            metadata={
                                "source": url,
                                "library": info["library"], # Required for Pie Chart
                                "type": info["type"]        # Required for Bar Chart
                            }
                        )
                        refined_docs.append(new_doc)
                print(f"✅ Successfully added {info['library']}")
            else:
                print(f"⚠️ Warning: No content returned for {info['library']}")
                
        except Exception as e:
            print(f"❌ Failed to scrape {info['library']}: {e}")

    # Final validation check for the terminal
    found_libs = set([d.metadata['library'] for d in refined_docs])
    required_libs = set([info['library'] for info in url_map.values()])
    missing = required_libs - found_libs
    
    if missing:
        print(f"🚨 ALERT: Missing libraries after scraping: {missing}")
    else:
        print("🎉 All libraries successfully scraped and labeled!")

    # Save data to local JSON for tracking
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw_json_data = [
        {
            "url": d.metadata["source"], 
            "library": d.metadata["library"], 
            "content_length": len(d.page_content)
        } 
        for d in refined_docs
    ]
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(raw_json_data, f, indent=4)
        
    return refined_docs

