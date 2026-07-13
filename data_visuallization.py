import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    # 1. Create target directories
    folders = ["static/plots", "public/plots"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}")
        
    # 2. Load dataset
    dataset_path = "flood_dataset.xlsx"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run generate_perfect_data.py first.")
        return
        
    df = pd.read_excel(dataset_path)
    
    # 3. Descriptive analysis log
    descriptive_log_path = "descriptive_analysis.txt"
    with open(descriptive_log_path, "w") as f:
        f.write("=========================================\n")
        f.write("      Rising Waters Data Exploration     \n")
        f.write("=========================================\n\n")
        f.write("--- FIRST 5 RECORDS (head) ---\n")
        f.write(df.head().to_string())
        f.write("\n\n--- SHAPE ---\n")
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
        f.write("\n--- DATA INFO ---\n")
        # Custom buffer write to get df.info() as string
        import io
        buf = io.StringIO()
        df.info(buf)
        f.write(buf.getvalue())
        f.write("\n\n--- STATISTICAL SUMMARY (describe) ---\n")
        f.write(df.describe().to_string())
        f.write("\n\n--- TARGET CLASS COUNTS ---\n")
        f.write(df["class"].value_counts().to_string())
        
    print(f"Generated descriptive log: {descriptive_log_path}")
    
    # Configure Seaborn style
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.facecolor': '#061826',
        'axes.facecolor': '#0d1f2d',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'axes.edgecolor': '#ffffff1a'
    })
    
    numeric_cols = ["Annual Rainfall", "Cloud Visibility", "Seasonal Rainfall", "Temperature", "Humidity"]
    
    # 4. Univariate Distribution Plots
    print("Generating distribution plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Univariate Feature Distributions", fontsize=16, color='#00E5FF', weight='bold')
    
    axes_flat = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes_flat[i], color='#4FC3F7', edgecolor='#ffffff26')
        axes_flat[i].set_title(col, fontsize=12, color='white')
        axes_flat[i].set_xlabel('')
        axes_flat[i].set_ylabel('Frequency')
        
    # Hide the empty 6th subplot
    axes_flat[5].set_visible(False)
    
    plt.tight_layout()
    for folder in folders:
        plt.savefig(os.path.join(folder, "distribution_plots.png"), dpi=150, facecolor='#061826')
    plt.close()
    
    # 5. Univariate Box Plots (Outlier Detection)
    print("Generating box plots for outlier detection...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Outlier Analysis via Box Plots", fontsize=16, color='#00E5FF', weight='bold')
    
    axes_flat = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes_flat[i], color='#FF5252')
        axes_flat[i].set_title(col, fontsize=12, color='white')
        axes_flat[i].set_ylabel('')
        
    axes_flat[5].set_visible(False)
    
    plt.tight_layout()
    for folder in folders:
        plt.savefig(os.path.join(folder, "box_plots.png"), dpi=150, facecolor='#061826')
    plt.close()
    
    # 6. Multivariate Correlation Heatmap
    print("Generating correlation heatmap...")
    # Map class to integer temporary for correlation
    df_corr = df.copy()
    df_corr["Cloud Cover"] = df_corr["Cloud Cover"].map({"Low": 0, "Medium": 1, "High": 2})
    df_corr["class"] = df_corr["class"].map({"Flood": 0, "No Flood": 1})
    
    plt.figure(figsize=(10, 8))
    corr = df_corr.corr()
    
    # Custom color palette matching Aqua Intelligence
    cmap = sns.diverging_palette(220, 180, as_cmap=True)
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 11, "weight": "bold"})
    
    plt.title("Multivariate Feature Correlation Heatmap", fontsize=14, color='#00E5FF', weight='bold', pad=20)
    plt.tight_layout()
    
    for folder in folders:
        plt.savefig(os.path.join(folder, "correlation_heatmap.png"), dpi=150, facecolor='#061826')
    plt.close()
    
    print("Visualizations generated and saved successfully!")

if __name__ == "__main__":
    generate_visualizations()
