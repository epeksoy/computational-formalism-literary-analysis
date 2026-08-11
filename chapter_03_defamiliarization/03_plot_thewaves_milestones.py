import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_milestone_plot():
    # 1. Load Data
    novel_name = "TheWaves"
    output_dir = "results"
    
    csv_path = os.path.join(output_dir, f"{novel_name}_Sentence_Data.csv")
    eltec_path = os.path.join(output_dir, "ELTeC_Baseline_Sentence_Data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    sentences_df = pd.read_csv(csv_path)
    
    baseline_mean = 0
    if os.path.exists(eltec_path):
        baseline_df = pd.read_csv(eltec_path)
        baseline_mean = baseline_df['Perplexity'].mean()

    # 2. Define Milestones
    milestones = {
        12: "SUNRISE: EARLY CHILDHOOD",
        389: "MID-MORNING: CHILDHOOD AND EARLY SCHOOL YEARS",
        1175: "LATE MORNING: ADOLESCENCE AND UNIVERSITY",
        1774: "NOONDAY: EARLY ADULTHOOD AND ESTABLISHING CAREERS",
        2300: "MID-AFTERNOON: THE DEATH OF PERCIVAL",
        2581: "EARLY AFTERNOON: MID-ADULTHOOD",
        3193: "LATE AFTERNOON: LATE ADULTHOOD AND LOOKING BACK",
        3603: "NIGHT/SUNSET: DEATH AND THE END OF THE DAY",
        4156: "EVENING: APPROACHING OLD AGE"
    }

    # 3. Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8)) 
    
    # Calculate rolling mean
    rolling_ppl = sentences_df['Perplexity'].rolling(window=50, min_periods=1).mean()
    
    # Plot line
    plt.plot(sentences_df['Sentence_Number'], rolling_ppl,
             color='darkorange', linewidth=2, label='Rolling mean (w=50)')
             
    # Plot baseline
    if baseline_mean > 0:
        plt.axhline(y=baseline_mean, color='steelblue', linestyle='--',
                    label='ELTeC baseline mean')

    # Add Milestones
    y_max = rolling_ppl.max()
    y_min = rolling_ppl.min()
    
    for idx, name in milestones.items():
        # Draw vertical line
        plt.axvline(x=idx, color='gray', linestyle=':', alpha=0.6)
        # Draw text rotated 90 degrees
        plt.text(idx - 20, y_max + 1, name, rotation=90, va='top', ha='right', 
                 fontsize=10, color='black', fontweight='bold', alpha=0.8)

    plt.title(f'Narrative Milestones & Rolling Perplexity: {novel_name}', fontsize=16, pad=20)
    plt.xlabel('Sentence Number', fontsize=14)
    plt.ylabel('Rolling Average Perplexity', fontsize=14)
    
    # Extend y-axis slightly so text fits without overlapping the title
    plt.ylim(bottom=max(0, y_min - 5), top=y_max + 5)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save
    out_file = os.path.join(output_dir, f"{novel_name}_Figure5_Milestones.png")
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")
    plt.close()

if __name__ == "__main__":
    generate_milestone_plot()
