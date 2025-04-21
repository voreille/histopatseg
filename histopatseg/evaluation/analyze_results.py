import json
from pathlib import Path
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@click.command()
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the results pickle file or directory with results.",
)
@click.option(
    "--compare-with",
    multiple=True,
    type=click.Path(exists=True),
    help="Path to additional results to compare with (can be used multiple times).",
)
@click.option(
    "--metrics",
    default="f1_macro,balanced_accuracy",
    help="Comma-separated list of metrics to analyze.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save analysis results. Defaults to same directory as results.",
)
def analyze(results_path, compare_with, metrics, output_dir):
    """Analyze and visualize model evaluation results"""
    results_path = Path(results_path)
    
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    all_results = []
    
    # Function to load and process a single result file
    def load_result(path):
        with open(path, 'rb') as f:
            result = pickle.load(f)
            # Extract key metadata for comparison
            config = result['config']
            model_name = config.get('model_name', 'unknown')
            task = config.get('task', 'unknown')
            superclass = config.get('superclass_to_keep', 'all')
            agg = config.get('aggregation_method', 'none')
            
            # Create a short name for plots
            short_name = f"{model_name}"
            if superclass != 'all':
                short_name += f"_{superclass}"
            if agg != 'none':
                short_name += f"_{agg}"
            
            result['short_name'] = short_name
            return result
    
    # Load primary result
    if results_path.is_file():
        all_results.append(load_result(results_path))
    else:
        # If a directory, load all pickle files
        for pkl_file in results_path.glob('*_results.pkl'):
            all_results.append(load_result(pkl_file))
    
    # Load comparison results
    for compare_path in compare_with:
        compare_path = Path(compare_path)
        if compare_path.is_file():
            all_results.append(load_result(compare_path))
        elif compare_path.is_dir():
            for pkl_file in compare_path.glob('*_results.pkl'):
                all_results.append(load_result(pkl_file))
    
    # Split metrics string into list
    metrics_list = [m.strip() for m in metrics.split(',')]
    
    # Compare key metrics across experiments
    compare_metrics(all_results, metrics_list, output_dir)
    
    # Generate confusion matrix comparisons
    compare_confusion_matrices(all_results, output_dir)
    
    # Compare per-class performance
    compare_class_performance(all_results, output_dir)
    
    # If there's just one result, do additional in-depth analysis
    if len(all_results) == 1:
        analyze_single_result(all_results[0], output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


def compare_metrics(all_results, metrics_list, output_dir):
    """Compare specified metrics across all results"""
    plt.figure(figsize=(12, len(metrics_list) * 4))
    
    for i, metric in enumerate(metrics_list):
        plt.subplot(len(metrics_list), 1, i+1)
        
        # Prepare data for plotting
        data = []
        for result in all_results:
            for clf in ['knn', 'linear']:
                mean_key = f"{clf}_{metric}_mean"
                std_key = f"{clf}_{metric}_std"
                
                if mean_key in result['summary']:
                    data.append({
                        'Model': result['short_name'],
                        'Classifier': clf.upper(),
                        'Score': result['summary'][mean_key],
                        'Std': result['summary'].get(std_key, 0)
                    })
        
        # Create DataFrame for plotting
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Plot bars with error bars
            ax = sns.barplot(x='Model', y='Score', hue='Classifier', data=df, palette='Set2')
            
            # Add error bars
            for j, row in df.iterrows():
                ax.errorbar(
                    j % len(all_results) + (0 if row['Classifier'] == 'KNN' else 0.25),
                    row['Score'],
                    yerr=row['Std'],
                    fmt='none',
                    c='black',
                    capsize=5
                )
            
            plt.title(f"{metric.replace('_', ' ').title()} Comparison")
            plt.ylim(0, 1.05)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_confusion_matrices(all_results, output_dir):
    """Compare confusion matrices across experiments"""
    # Implementation details...
    pass


def compare_class_performance(all_results, output_dir):
    """Compare per-class metrics across experiments"""
    # Implementation details...
    pass


def analyze_single_result(result, output_dir):
    """Perform in-depth analysis on a single result"""
    # Implementation details...
    pass


if __name__ == '__main__':
    analyze()