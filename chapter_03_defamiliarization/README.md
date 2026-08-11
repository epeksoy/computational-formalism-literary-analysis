# Chapter 2 - Defamiliarization with GPT-2

This project operationalises Viktor Shklovsky's concept of defamiliarization (*ostranenie*) in a computational framework. It fine-tunes a GPT-2 language model on the European Literary Text Collection (ELTeC) corpus of historical novels to establish a stylometric baseline of "familiar" 19th-century prose. It then measures how much specific test novels (like *Oliver Twist*, *The Waves*, and *Ulysses*) depart from that baseline via perplexity scores.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install requirements:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Requirements (ELTeC Corpus):**
   The ELTeC dataset is not included in this repository due to size. To run the fine-tuning, you must download the ELTeC text corpus and place the `.txt` files in a folder named `ELTEC` in the root of the project directory. 
   - You can access the ELTeC collections via their official repositories (e.g., [COST Action Distant Reading](https://www.distant-reading.net/eltec/)).

4. **Test Novels:**
   The `test_novels/` folder contains the specific literary works that will be evaluated against the ELTeC baseline model.

## Usage

### 1. Run the Main Analysis
The primary script handles fine-tuning the model (or loading it if it already exists) and computing the sentence-level and document-level perplexity.

```bash
python defamiliarization_analysis.py
```
*Note: You can adjust the `MODE` variable inside `defamiliarization_analysis.py` to `FAST`, `FASTER`, or `FULL` depending on how many epochs and sentences you want to process.*

Outputs, including CSV data and PNG charts, will be generated in the `output_defamiliarization/` directory.

### 2. Run the Plotting Scripts
Once the main analysis has generated the `*_Sentence_Data.csv` files in the output directory, you can run the milestone plotting scripts to visualize specific textual milestones in the novels:

```bash
python plot_oliver_twist_milestones.py
python plot_thewaves_milestones.py
python plot_ulysses_milestones.py
```
