# Computational Formalism & Literary Analysis

This repository contains the data, scripts, and computational pipelines used for analyzing literature through various formalist lenses. The project is divided into three main chapters, each applying distinct natural language processing (NLP) and machine learning techniques to specific literary corpora.

---

## Repository Structure

The project is divided into three self-contained chapter directories. Each directory contains its own processing scripts, specific `README.md` files with detailed instructions, and a `results/` folder for outputs. 

*Note: Large datasets (such as ELTeC and CONLIT) and generated models/embeddings are **not included** in this repository due to size constraints. The corresponding directories are kept intentionally empty (via `.gitkeep` files) so you can download and place the data into the correct locations.*

### [Chapter 3: Defamiliarization](chapter_03_defamiliarization/README.md)
This chapter operationalizes Viktor Shklovsky's concept of *defamiliarization* (ostranenie) in a computational framework. 
- **Method:** Fine-tunes a GPT-2 language model on the European Literary Text Collection (ELTeC) corpus of historical novels to establish a stylometric baseline of "familiar" 19th-century prose. It then measures how much specific test novels (like *Oliver Twist*, *The Waves*, and *Ulysses*) depart from that baseline via perplexity scores.
- **Data required:** ELTeC corpus.

### [Chapter 4: Literariness](chapter_04_literariness/README.md)
This chapter focuses on processing and analyzing the "Literariness" of texts using the CONLIT dataset.
- **Method:** Uses classifiers and statistical profiling (univariate discrimination, PCA, etc.) to examine genre boundaries, testing orthodoxy and prestige in contemporary literature.
- **Data required:** CONLIT dataset (Contemporary Literature), compiled by the txtLAB at McGill University.

### [Chapter 5: Syuzhet](chapter_05_syuzhet/README.md)
This chapter analyzes narrative arcs across a large corpus of books.
- **Method:** Provides a pipeline to download books from Project Gutenberg, compute their semantic narrative arcs using SBERT (Sentence-BERT), cluster them, and statistically validate the extracted narrative shapes.

---

## General Setup

To run the analyses in this repository, it is recommended to use a virtual environment. You will need to install the dependencies required for each chapter. 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/epeksoy/computational-formalism-literary-analysis.git
   cd computational-formalism-literary-analysis
   ```

2. **Install requirements:**
   Each chapter folder contains a specific `requirements.txt`. Navigate to the chapter you wish to run and install its dependencies.
   ```bash
   cd chapter_03_defamiliarization
   pip install -r requirements.txt
   ```

3. **Provide the Data:**
   Please read the `README.md` inside the specific chapter folder for detailed instructions on where to acquire the necessary corpus datasets (e.g., ELTeC, CONLIT) and where to place them before running the scripts.
