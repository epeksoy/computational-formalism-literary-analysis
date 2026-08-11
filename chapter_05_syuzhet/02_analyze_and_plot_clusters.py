import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
import argparse

# Suppress SciPy ClusterWarning regarding redundant distance matrix
from scipy.cluster.hierarchy import ClusterWarning
warnings.simplefilter("ignore", ClusterWarning)

# ===================== CONFIG =====================
parser = argparse.ArgumentParser(description="Analyze narrative arcs and generate SVD clusters.")
parser.add_argument("-k", type=int, default=6, help="Number of clusters for SVD decomposition (default: 6)")
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
ARC_DIR = os.path.join(BASE_DIR, "arcs_global")
ARC_TABLE = os.path.join(BASE_DIR, "arc_vectors_global.parquet")
ARC_TABLE_CSV = os.path.join(BASE_DIR, "arc_vectors_global.csv")
META_FILE_ARC = os.path.join(BASE_DIR, "processed_arcs_global.json")
GLOBAL_CENTROID_FILE = os.path.join(BASE_DIR, "global_centroid.npy")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "results")


def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(ARC_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# PHASE 1: NARRATIVE ARC EXTRACTION FUNCTIONS
# -------------------------------------------------------------------------
def calculate_global_centroid(emb_files):
    print("Computing global centroid across all novels...")
    sum_embeddings = None
    total_chunks = 0

    for i, emb_path in enumerate(emb_files):
        try:
            embeddings = np.load(emb_path)
            if embeddings.shape[0] == 0:
                continue
            
            chunk_sum = np.sum(embeddings, axis=0)
            if sum_embeddings is None:
                sum_embeddings = chunk_sum
            else:
                sum_embeddings += chunk_sum
                
            total_chunks += embeddings.shape[0]
            
            if (i + 1) % 100 == 0:
                print(f"  -> Processed {i + 1}/{len(emb_files)} files...")
        except Exception as e:
            print(f"  -> Error reading {emb_path}: {e}")

    if total_chunks == 0:
        raise ValueError("No embeddings found or successfully loaded to compute global centroid.")

    global_centroid = sum_embeddings / total_chunks
    np.save(GLOBAL_CENTROID_FILE, global_centroid)
    print(f"Global centroid calculated from {total_chunks} total chunks and saved to {GLOBAL_CENTROID_FILE}")
    return global_centroid

def compute_arc_from_global_centroid(embeddings, global_centroid):
    centroid_2d = global_centroid.reshape(1, -1)
    sims = cosine_similarity(embeddings, centroid_2d).flatten()
    distances = 1 - sims

    if len(distances) <= 1:
        return np.interp(np.linspace(0, 100, 100), [0, 100], [distances[0], distances[-1]])

    window_size = max(10, len(distances) // 10)
    if window_size < 2:
        smoothed = distances
    else:
        smoothed = np.convolve(distances, np.ones(window_size)/window_size, mode="valid")

    x_smoothed = np.linspace(0, 100, len(smoothed))
    x_target = np.linspace(0, 100, 100)
    arc_100 = np.interp(x_target, x_smoothed, smoothed)
    return arc_100

def append_arc_row(filename, arc_vector, num_chunks):
    row = {"filename": filename, "num_chunks": num_chunks}
    for i, val in enumerate(arc_vector):
        row[f"point_{i}"] = float(val)

    df_new = pd.DataFrame([row])

    if ARC_TABLE.endswith(".parquet"):
        if os.path.exists(ARC_TABLE):
            df_existing = pd.read_parquet(ARC_TABLE)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
            df_all.to_parquet(ARC_TABLE, index=False)
        else:
            df_new.to_parquet(ARC_TABLE, index=False)
    else:
        if os.path.exists(ARC_TABLE_CSV):
            df_new.to_csv(ARC_TABLE_CSV, mode="a", header=False, index=False)
        else:
            df_new.to_csv(ARC_TABLE_CSV, index=False)

def load_processed():
    if os.path.exists(META_FILE_ARC):
        with open(META_FILE_ARC, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed(processed_set):
    with open(META_FILE_ARC, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_set)), f, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------------
# PHASE 2 & 3: DENDROGRAMS & SVD CLUSTERING FUNCTIONS
# -------------------------------------------------------------------------
def load_datasets():
    if not os.path.exists(METADATA_FILE):
        print(f"Warning: Metadata file not found at: {METADATA_FILE}")
        df_meta = pd.DataFrame(columns=['filename'])
    else:
        try:
            df_meta = pd.read_csv(METADATA_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_meta = pd.read_csv(METADATA_FILE, encoding='latin-1')
            except UnicodeDecodeError:
                df_meta = pd.read_csv(METADATA_FILE, encoding='cp1252')
                
        df_meta.columns = df_meta.columns.str.strip()
        if 'filename' not in df_meta.columns and 'gutenberg_id' in df_meta.columns:
            df_meta['filename'] = df_meta['gutenberg_id'].astype(str).str.strip() + ".txt"
        elif 'filename' in df_meta.columns:
            df_meta['filename'] = df_meta['filename'].astype(str).str.strip()
        if 'filename' in df_meta.columns:
            df_meta = df_meta.drop_duplicates(subset=["filename"])
            
    if os.path.exists(ARC_TABLE):
        df_arcs = pd.read_parquet(ARC_TABLE)
    elif os.path.exists(ARC_TABLE_CSV):
        df_arcs = pd.read_csv(ARC_TABLE_CSV)
    else:
        raise FileNotFoundError("Global arc table not found. Ensure Phase 1 runs correctly.")
        
    df_arcs.columns = df_arcs.columns.str.strip()
    df_arcs['filename'] = df_arcs['filename'].astype(str).str.strip()
    df_arcs = df_arcs.drop_duplicates(subset=["filename"])
    
    merged_df = pd.merge(df_arcs, df_meta, on="filename", how="left")
    return merged_df

def plot_cluster_averages(df, X, k, cluster_names):
    x = np.linspace(0, 100, 100)
    plt.figure(figsize=(12, 8), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, k)) if k <= 10 else plt.cm.Set3(np.linspace(0, 1, k))
    
    for idx, cid in enumerate(sorted(df['cluster'].unique())):
        mask = df['cluster'] == cid
        cluster_arcs = X[mask]
        if len(cluster_arcs) == 0:
            continue
        mean_arc = np.mean(cluster_arcs, axis=0)
        n_books = len(cluster_arcs)
        name = cluster_names.get(cid, f"Cluster {cid}")
        plt.plot(x, mean_arc, label=f"{name} (n={n_books})", linewidth=3.0, color=colors[idx % len(colors)])
        
    plt.title(f"Average Narrative Arcs per SVD Archetype (K={k})", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Narrative Progression (%)", fontsize=12, fontweight='semibold')
    plt.ylabel("Normalized Cosine Distance from Global Centroid", fontsize=12, fontweight='semibold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "average_arc_per_cluster_global.png"), dpi=300)
    plt.close()

def plot_individual_clusters_shape(df, X, cluster_names):
    x = np.linspace(0, 100, 100)
    print("Generating individual cluster shape profiles...")
    for cid in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cid
        cluster_arcs = X[mask]
        if len(cluster_arcs) == 0:
            continue
        cluster_filenames = df['filename'][mask].tolist()
        n_books = len(cluster_arcs)
        name = cluster_names.get(cid, f"Cluster {cid}")
        
        mean_arc = np.mean(cluster_arcs, axis=0)
        std_arc = np.std(cluster_arcs, axis=0)
        
        mean_arc_norm = (mean_arc - mean_arc.mean()) / (mean_arc.std() + 1e-8)
        cluster_arcs_norm = (cluster_arcs - cluster_arcs.mean(axis=1, keepdims=True)) / (cluster_arcs.std(axis=1, keepdims=True) + 1e-8)
        
        distances = np.linalg.norm(cluster_arcs_norm - mean_arc_norm, axis=1)
        closest_idx = np.argmin(distances)
        
        representative_filename = cluster_filenames[closest_idx]
        representative_arc = cluster_arcs[closest_idx]
        
        rep_row = df[mask].iloc[closest_idx]
        title = rep_row.get('title')
        author = rep_row.get('author')
        if pd.notna(title) and str(title).strip() != '' and str(title).lower() != 'nan':
            rep_label = f"Rep: {title} ({author})"
        else:
            rep_label = f"Rep: {representative_filename}"
            
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        for arc in cluster_arcs:
            ax.plot(x, arc, color='lightgray', alpha=0.07, linewidth=0.8)
        ax.plot(x, representative_arc, color='royalblue', linestyle='--', linewidth=1.8, label=rep_label)
        ax.plot(x, mean_arc, color='red', linewidth=3, label='Mean Shape')
        ax.fill_between(x, mean_arc - std_arc, mean_arc + std_arc, color='red', alpha=0.25)
        
        ax.set_title(f"SVD Archetype: {name}\n({n_books} books)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Narrative Progression (%)")
        ax.set_ylabel("Normalized Cosine Distance from Global Centroid")
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#ddd")
        ax.grid(True, alpha=0.3)
        
        safe_name = name.replace(" ", "_").replace("/", "_")
        output_name = f"{safe_name}_shape_detail.png"
        plt.savefig(os.path.join(VISUALS_DIR, output_name), dpi=300)
        plt.close()
        print(f" Saved Archetype ({name}) shape profile: {output_name}")

def find_best_representatives(df, X, cluster_names, n=3):
    from sklearn.metrics.pairwise import euclidean_distances
    print("\nPrototypical Representatives per Cluster:")
    for cid in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cid
        cluster_X = X[mask]
        if len(cluster_X) == 0:
            continue
        cluster_df = df[mask].reset_index(drop=True)
        name = cluster_names.get(cid, f"Cluster {cid}")
        
        mean_arc = np.mean(cluster_X, axis=0)
        mean_arc_norm = (mean_arc - mean_arc.mean()) / (mean_arc.std() + 1e-8)
        cluster_X_norm = (cluster_X - cluster_X.mean(axis=1, keepdims=True)) / (cluster_X.std(axis=1, keepdims=True) + 1e-8)
        distances = euclidean_distances(cluster_X_norm, mean_arc_norm.reshape(1, -1)).flatten()
        
        closest_idx = np.argsort(distances)[:n]
        print(f"\nArchetype: {name} ({len(cluster_df)} books):")
        for i, idx in enumerate(closest_idx):
            row = cluster_df.iloc[idx]
            dist = distances[idx]
            title = row.get('title')
            author = row.get('author')
            title_str = str(title) if pd.notna(title) and str(title).lower() != 'nan' else 'Unknown Title'
            author_str = str(author) if pd.notna(author) and str(author).lower() != 'nan' else 'Unknown Author'
            print(f" {i+1}. [{row['filename']}] {title_str} - {author_str} (dist={dist:.4f})")

def plot_svd_modes(Vt, cluster_names, m):
    x = np.linspace(0, 100, 100)
    fig, axes = plt.subplots(m, 1, figsize=(12, 3 * m), sharex=True, dpi=300)
    if m == 1:
        axes = [axes]
    for j in range(m):
        mode_vec = Vt[j, :]
        pos_name = cluster_names.get(j * 2, f"Mode {j+1} (+)")
        neg_name = cluster_names.get(j * 2 + 1, f"Mode {j+1} (-)")
        ax = axes[j]
        ax.plot(x, mode_vec, label=f"+SV{j+1}: {pos_name}", color='royalblue', linewidth=2.5)
        ax.plot(x, -mode_vec, label=f"-SV{j+1}: {neg_name}", color='indianred', linestyle='--', linewidth=2.0)
        ax.set_title(f"SVD Mode {j+1}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ddd")
        
    plt.suptitle("SVD Basis Modes & Literary Archetypes (Eigenmoods)", fontsize=16, fontweight='bold', y=0.98)
    plt.xlabel("Narrative Progression (%)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(VISUALS_DIR, "svd_modes_global.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved SVD basis modes visualization to: {output_path}")

def plot_dendrograms(X_centered, df):
    print("\nGenerating Ward hierarchical dendrograms...")
    dist_matrix = squareform(pdist(X_centered, metric='cityblock'))
    linked = linkage(dist_matrix, method='ward', metric='euclidean')
    
    # 1. Truncated Dendrogram
    plt.figure(figsize=(14, 8), dpi=300)
    dendrogram(linked, orientation="top", truncate_mode="lastp", p=35, show_contracted=True, leaf_font_size=10)
    plt.title(f"Ward's Hierarchical Clustering Dendrogram (Top 35 Shape Families)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Shape Family Leaves", fontsize=11, fontweight='semibold', labelpad=10)
    plt.ylabel("Variance Height", fontsize=11, fontweight='semibold', labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "dendrogram_ward_truncated.png"), dpi=300)
    plt.close()
    print("Saved dendrogram_ward_truncated.png")

    # 2. Full Dendrogram
    plt.figure(figsize=(15, 7), dpi=300)
    dendrogram(linked, orientation="top", no_labels=True, distance_sort="descending")
    plt.title(f"Ward's Hierarchical Clustering Dendrogram (Full Corpus Structure — {len(df)} Novels)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Novels", fontsize=11, fontweight='semibold', labelpad=10)
    plt.ylabel("Variance Height", fontsize=11, fontweight='semibold', labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "dendrogram_ward_full.png"), dpi=300)
    plt.close()
    print("Saved dendrogram_ward_full.png")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" STAGE 2: NARRATIVE ARC ANALYSIS & CLUSTERING ")
    print("=" * 60)

    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
    except ImportError:
        pass

    ensure_dirs()

    if not os.path.exists(EMB_DIR):
        raise FileNotFoundError(f"Embeddings directory not found: {EMB_DIR}")

    emb_files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
    if not emb_files:
        raise RuntimeError(f"No computed embedding (.npy) files found in {EMB_DIR}")

    # 1. Obtain Global Centroid
    if os.path.exists(GLOBAL_CENTROID_FILE):
        print(f"Loading cached global centroid from {GLOBAL_CENTROID_FILE} ...")
        global_centroid = np.load(GLOBAL_CENTROID_FILE)
    else:
        global_centroid = calculate_global_centroid(emb_files)

    # 2. Extract arcs relative to global centroid
    processed = load_processed()
    print(f"Total embeddings found: {len(emb_files)} | Already processed global arcs: {len(processed)}")

    for emb_path in emb_files:
        filename_npy = os.path.basename(emb_path)
        text_id = filename_npy[:-4]
        filename_txt = f"{text_id}.txt"
        arc_path = os.path.join(ARC_DIR, f"{text_id}_arc_global.npy")

        if filename_npy in processed and os.path.exists(arc_path):
            continue

        if os.path.exists(arc_path):
            print(f"  -> [GLOBAL ARC CACHE HIT] loading precomputed global arc for {filename_npy}")
            arc = np.load(arc_path)
            embeddings = np.load(emb_path)
            num_chunks = embeddings.shape[0]
        else:
            print(f"Extracting global narrative arc from {filename_npy} ...")
            embeddings = np.load(emb_path)
            if len(embeddings) < 2:
                processed.add(filename_npy)
                save_processed(processed)
                continue
            arc = compute_arc_from_global_centroid(embeddings, global_centroid)
            np.save(arc_path, arc)
            num_chunks = embeddings.shape[0]

        append_arc_row(filename_txt, arc, num_chunks=num_chunks)
        processed.add(filename_npy)
        save_processed(processed)

    print("\nAll narrative arcs processed. Loading arc table for clustering...")
    try:
        df = load_datasets()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    if len(df) < 5:
        print("Too few texts for meaningful clustering.")
        return

    N = len(df)
    X = df[[f"point_{i}" for i in range(100)]].to_numpy().astype(np.float64)
    X_centered = X - X.mean(axis=1, keepdims=True)

    # 3. Dendrograms
    plot_dendrograms(X_centered, df)

    # 4. SVD Clustering
    k = args.k
    print("=" * 60)
    print(f" CREATING {k} CORPUS CLUSTERS WITH SVD DECOMPOSITION ")
    print("=" * 60)
    if k % 2 != 0:
        print(f"WARNING: Adjusting K from {k} to {k+1} to support SVD polarity.")
        k += 1
    m = k // 2
    
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = X_centered @ Vt[:m].T
    
    best_mode = np.argmax(np.abs(W), axis=1)
    best_polarity = np.sign(W[np.arange(N), best_mode])
    labels = best_mode * 2 + (best_polarity < 0).astype(int)
    df['cluster'] = labels
    
    cluster_names = {}
    for j in range(m):
        mode_vec = Vt[j, :]
        if j == 0:
            trend = mode_vec[-1] - mode_vec[0]
            if trend > 0:
                pos_label, neg_label = "1. Rags to Riches", "2. Riches to Rags"
            else:
                pos_label, neg_label = "2. Riches to Rags", "1. Rags to Riches"
        elif j == 1:
            middle_val = mode_vec[25:75].mean()
            ends_val = (mode_vec[:25].mean() + mode_vec[75:].mean()) / 2
            if middle_val < ends_val:
                pos_label, neg_label = "3. Man in a Hole", "4. Icarus"
            else:
                pos_label, neg_label = "4. Icarus", "3. Man in a Hole"
        elif j == 2:
            if mode_vec[:50].mean() > mode_vec[50:].mean():
                pos_label, neg_label = "5. Cinderella", "6. Oedipus"
            else:
                pos_label, neg_label = "6. Oedipus", "5. Cinderella"
        else:
            pos_label, neg_label = f"Mode {j+1} (+)", f"Mode {j+1} (-)"
            
        cluster_names[j * 2] = pos_label
        cluster_names[j * 2 + 1] = neg_label
        
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    plot_svd_modes(Vt, cluster_names, m)
    
    print("\nUpdating UMAP coordinates...")
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        X_umap = reducer.fit_transform(W)
        df['umap_x'] = X_umap[:, 0]
        df['umap_y'] = X_umap[:, 1]
        
        plt.figure(figsize=(12, 9), dpi=300)
        unique_clusters = sorted(df['cluster'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        for idx, cid in enumerate(unique_clusters):
            mask = df['cluster'] == cid
            name = cluster_names.get(cid, f"Cluster {cid}")
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=name, color=colors[idx % len(colors)], s=25, alpha=0.8)
        plt.legend(title="SVD Archetypes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"UMAP Projection of SVD Syuzhet Plot Clusters (K={k})", fontsize=14, fontweight='bold')
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        
        # Prevent extreme outliers from squishing the plot
        x_min, x_max = np.percentile(X_umap[:, 0], [1, 99])
        y_min, y_max = np.percentile(X_umap[:, 1], [1, 99])
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, "umap_clusters_global.png"), dpi=300)
        plt.close()
    except ImportError:
        print(" UMAP not installed. Skipping coordinate updates.")
        
    plot_cluster_averages(df, X, k, cluster_names)
    plot_individual_clusters_shape(df, X, cluster_names)
    find_best_representatives(df, X, cluster_names, n=3)
    
    df_out = df.drop_duplicates(subset=["filename"])
    output_csv = os.path.join(BASE_DIR, "arc_vectors_clustered_global_enriched.csv")
    original_cols = ["filename", "num_chunks", "cluster", "cluster_name", "umap_x", "umap_y"]
    available_cols = [c for c in original_cols if c in df_out.columns] + [f"point_{i}" for i in range(100)]
    df_save = df_out[available_cols]
    df_save.to_csv(output_csv, index=False)
    print(f"\nSuccessfully generated {k} clusters and saved to: {output_csv}")

if __name__ == "__main__":
    main()
