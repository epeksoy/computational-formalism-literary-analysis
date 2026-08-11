import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
parser = argparse.ArgumentParser(description="Plot and compare global reference narrative arcs of individual novels.")
parser.add_argument("--ids", type=str, help="Comma-separated Gutenberg IDs (e.g., '11,12')")
parser.add_argument("--titles", type=str, help="Comma-separated Title keywords (e.g., 'Alice,Peter')")
parser.add_argument("--authors", type=str, help="Comma-separated Author keywords (e.g., 'Carroll,Wells')")
parser.add_argument("--output", type=str, default="comparison_plot.png", help="Filename of the saved plot")
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
ARC_TABLE = os.path.join(BASE_DIR, "arc_vectors_global.parquet")
ARC_TABLE_CSV = os.path.join(BASE_DIR, "arc_vectors_global.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(VISUALS_DIR, exist_ok=True)


def load_datasets():
    # 1. Load metadata
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found at: {METADATA_FILE}")
    
    # Try different encodings for metadata.csv
    try:
        df_meta = pd.read_csv(METADATA_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_meta = pd.read_csv(METADATA_FILE, encoding='latin-1')
        except UnicodeDecodeError:
            df_meta = pd.read_csv(METADATA_FILE, encoding='cp1252')
    
    # Clean up column names and string fields
    df_meta.columns = df_meta.columns.str.strip()
    if 'filename' not in df_meta.columns and 'gutenberg_id' in df_meta.columns:
        df_meta['filename'] = df_meta['gutenberg_id'].astype(str).str.strip() + ".txt"
    elif 'filename' in df_meta.columns:
        df_meta['filename'] = df_meta['filename'].astype(str).str.strip()
    
    # Load arc vectors (Global Arcs)
    if os.path.exists(ARC_TABLE):
        df_arcs = pd.read_parquet(ARC_TABLE)
    elif os.path.exists(ARC_TABLE_CSV):
        df_arcs = pd.read_csv(ARC_TABLE_CSV)
    else:
        raise FileNotFoundError(f"Global arc table not found at {ARC_TABLE} or {ARC_TABLE_CSV}. Please run 02_analyze_and_plot_clusters.py first.")
        
    df_arcs.columns = df_arcs.columns.str.strip()
    df_arcs['filename'] = df_arcs['filename'].astype(str).str.strip()
    
    # Remove duplicates from arcs to avoid multiple matches
    df_arcs = df_arcs.drop_duplicates(subset=["filename"])
    
    # Merge metadata with arcs
    merged_df = pd.merge(df_meta, df_arcs, on="filename", how="inner")
    return merged_df


def search_books(df, ids=None, titles=None, authors=None):
    matched_indices = set()
    
    # 1. Search by Gutenberg ID
    if ids:
        id_list = [str(i).strip() for i in ids]
        df_ids = df[df['gutenberg_id'].astype(str).str.strip().isin(id_list)]
        matched_indices.update(df_ids.index)
        
    # 2. Search by Title (substring match, case insensitive)
    if titles:
        for t in titles:
            df_title = df[df['title'].astype(str).str.contains(t, case=False, na=False)]
            matched_indices.update(df_title.index)
            
    # 3. Search by Author (substring match, case insensitive)
    if authors:
        for a in authors:
            df_author = df[df['author'].astype(str).str.contains(a, case=False, na=False)]
            matched_indices.update(df_author.index)
            
    return df.loc[list(matched_indices)]


def plot_arcs(matched_df, output_path=None):
    if matched_df.empty:
        print("No matching books with computed arcs were found.")
        return
    
    if not output_path or os.path.basename(output_path) == "comparison_plot.png":
        titles = matched_df['title'].astype(str).tolist()
        short_titles = []
        for t in titles[:3]:
            clean_t = "".join(c if c.isalnum() else " " for c in t).split()
            if clean_t:
                short_titles.append(clean_t[0][:15])
        
        dynamic_name = "comparison_" + "_".join(short_titles) if short_titles else "comparison"
        if len(titles) > 3:
            dynamic_name += "_and_others"
        dynamic_name += ".png"
        
        dir_name = os.path.dirname(output_path) if output_path else VISUALS_DIR
        output_path = os.path.join(dir_name or VISUALS_DIR, dynamic_name)

    print(f"\nFound {len(matched_df)} books to plot:")
    for idx, row in matched_df.iterrows():
        print(f" - [{row['gutenberg_id']}] {row['title']} by {row['author']} ({row['filename']})")
        
    # Create the plot using a premium design aesthetic
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    # Palette of harmonious, vibrant colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    x_axis = np.linspace(0, 100, 100)
    
    for i, (_, row) in enumerate(matched_df.iterrows()):
        color = colors[i % len(colors)]
        # Retrieve the 100 point values
        arc_vector = row[[f"point_{j}" for j in range(100)]].to_numpy().astype(float)
        
        # Format label: "Title (Author)"
        label = f"{row['title']} - {row['author']}"
        if len(label) > 50:
            label = label[:47] + "..."
        label = f"[{row['gutenberg_id']}] {label}"
        
        ax.plot(x_axis, arc_vector, label=label, color=color, linewidth=2.5, alpha=0.9)
        
    ax.set_title("Narrative Arc Comparison (Global Reference Trajectory)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Narrative Progression (%)", fontsize=11, fontweight='semibold', labelpad=10)
    ax.set_ylabel("Cosine Distance from Global Centroid", fontsize=11, fontweight='semibold', labelpad=10)
    
    # Legend settings for readability
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#ddd", framealpha=0.9, fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\nSuccess! Plotted {len(matched_df)} arcs and saved comparison to: {output_path}")


def interactive_prompt(df):
    print("\n" + "=" * 60)
    print("      INTERACTIVE NARRATIVE ARC COMPARISON TOOL")
    print("=" * 60)
    print("Enter search queries to compare specific books (Global Reference Arcs).")
    
    while True:
        ids = input("\nEnter Gutenberg IDs (e.g., 11, 12, 16) [Enter to skip]: ").strip()
        titles = input("Enter Title keywords (e.g., Alice, Peter) [Enter to skip]: ").strip()
        authors = input("Enter Author names (e.g., Carroll, Barrie) [Enter to skip]: ").strip()
        
        id_list = [i.strip() for i in ids.split(",")] if ids else None
        title_list = [t.strip() for t in titles.split(",")] if titles else None
        author_list = [a.strip() for a in authors.split(",")] if authors else None
        
        if not id_list and not title_list and not author_list:
            print("Please provide at least one search query.")
            continue
            
        matched = search_books(df, id_list, title_list, author_list)
        
        if matched.empty:
            print("\nNo books matched your query. Let's try again!")
            continue
            
        plot_arcs(matched, output_path=None)
        break


def main():
    try:
        df = load_datasets()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # If no CLI args are passed, go to interactive mode
    if not args.ids and not args.titles and not args.authors:
        interactive_prompt(df)
    else:
        id_list = [i.strip() for i in args.ids.split(",")] if args.ids else None
        title_list = [t.strip() for t in args.titles.split(",")] if args.titles else None
        author_list = [a.strip() for a in args.authors.split(",")] if args.authors else None
        
        matched = search_books(df, id_list, title_list, author_list)
        out_path = os.path.join(VISUALS_DIR, args.output) if args.output != "comparison_plot.png" else None
        plot_arcs(matched, output_path=out_path)


if __name__ == "__main__":
    main()
