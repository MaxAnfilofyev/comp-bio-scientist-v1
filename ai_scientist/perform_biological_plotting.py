"""
Biological plotting and analysis tools for AI Scientist.
Provides visualization methods and statistical analysis frameworks
specific to computational biology research.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


class BiologicalPlotter:
    """Class for creating biological-specific visualizations"""

    def __init__(self, working_dir="./working"):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_pathway_network(self, pathway_data, title="Biological Pathway"):
        """Create pathway network visualization"""
        try:
            G = nx.Graph()

            # Add nodes and edges from pathway data
            if 'nodes' in pathway_data:
                for node in pathway_data['nodes']:
                    G.add_node(node['id'], **node.get('attributes', {}))

            if 'edges' in pathway_data:
                for edge in pathway_data['edges']:
                    G.add_edge(edge['source'], edge['target'],
                             **edge.get('attributes', {}))

            plt.figure(figsize=(12, 8))

            # Use spring layout for pathway visualization
            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Draw the network
            nx.draw(G, pos,
                   with_labels=True,
                   node_color='lightblue',
                   node_size=300,
                   font_size=8,
                   font_weight='bold',
                   edge_color='gray',
                   width=1.5,
                   alpha=0.7)

            plt.title(f"{title} - Network Visualization", fontsize=14)
            plt.axis('off')

            filepath = os.path.join(self.working_dir, "biological_pathway_network.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating pathway network plot: {e}")
            plt.close()
            return None

    def plot_gene_expression_heatmap(self, expression_data, gene_names=None,
                                   sample_labels=None, title="Gene Expression"):
        """Create gene expression heatmap"""
        try:
            plt.figure(figsize=(12, 8))

            # Create heatmap
            sns.heatmap(expression_data,
                       cmap='RdYlBu_r',
                       center=0,
                       xticklabels=sample_labels if sample_labels else True,
                       yticklabels=gene_names if gene_names else True,
                       cbar_kws={'label': 'Expression Level'})

            plt.title(f"{title} - Heatmap", fontsize=14)
            plt.xlabel("Samples")
            plt.ylabel("Genes")

            filepath = os.path.join(self.working_dir, "gene_expression_heatmap.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating gene expression heatmap: {e}")
            plt.close()
            return None

    def plot_volcano_plot(self, fold_changes, p_values, gene_names=None,
                         title="Differential Expression Analysis"):
        """Create volcano plot for differential expression analysis"""
        try:
            plt.figure(figsize=(10, 8))

            # Convert p-values to -log10 scale
            neg_log_p = -np.log10(p_values)

            # Create scatter plot
            plt.scatter(fold_changes, neg_log_p, alpha=0.6, s=20)

            # Add threshold lines
            plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                       label='p = 0.05')
            plt.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='FC = 2')
            plt.axvline(x=-1, color='blue', linestyle='--', alpha=0.7)

            # Label significantly differentially expressed genes
            if gene_names is not None:
                significant_up = (fold_changes > 1) & (p_values < 0.05)
                significant_down = (fold_changes < -1) & (p_values < 0.05)

                if np.any(significant_up):
                    plt.scatter(fold_changes[significant_up], neg_log_p[significant_up],
                              color='red', s=30, alpha=0.8)
                if np.any(significant_down):
                    plt.scatter(fold_changes[significant_down], neg_log_p[significant_down],
                              color='blue', s=30, alpha=0.8)

            plt.xlabel("log2(Fold Change)")
            plt.ylabel("-log10(p-value)")
            plt.title(f"{title} - Volcano Plot", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            filepath = os.path.join(self.working_dir, "differential_expression_volcano.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating volcano plot: {e}")
            plt.close()
            return None

    def plot_protein_structure_metric(self, structure_metrics, metric_name="RMSD",
                                    title="Protein Structure Analysis"):
        """Plot protein structure quality metrics"""
        try:
            plt.figure(figsize=(10, 6))

            if 'predicted' in structure_metrics and 'reference' in structure_metrics:
                # Plot comparison
                x = range(len(structure_metrics['predicted']))
                plt.plot(x, structure_metrics['predicted'], 'o-', label='Predicted',
                        color='blue', alpha=0.7)
                plt.plot(x, structure_metrics['reference'], 's-', label='Reference',
                        color='red', alpha=0.7)
            else:
                # Single metric plot
                plt.plot(structure_metrics, 'o-', color='green', alpha=0.7)

            plt.xlabel("Residue Position" if 'predicted' in structure_metrics else "Sample")
            plt.ylabel(metric_name)
            plt.title(f"{title} - {metric_name}", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            filepath = os.path.join(self.working_dir, f"protein_structure_{metric_name.lower()}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating protein structure plot: {e}")
            plt.close()
            return None

    def plot_drug_target_interactions(self, interaction_scores, drug_names=None,
                                    target_names=None, title="Drug-Target Interactions"):
        """Create drug-target interaction heatmap"""
        try:
            plt.figure(figsize=(12, 8))

            # Create interaction matrix
            sns.heatmap(interaction_scores,
                       cmap='viridis',
                       xticklabels=target_names if target_names else True,
                       yticklabels=drug_names if drug_names else True,
                       cbar_kws={'label': 'Interaction Score'})

            plt.title(f"{title} - Interaction Matrix", fontsize=14)
            plt.xlabel("Targets")
            plt.ylabel("Drugs")

            filepath = os.path.join(self.working_dir, "drug_target_interactions.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating drug-target interaction plot: {e}")
            plt.close()
            return None

    def plot_roc_curve_biological(self, y_true, y_scores, labels=None,
                                title="Biological Classification Performance"):
        """Plot ROC curve for biological classification tasks"""
        try:
            plt.figure(figsize=(8, 8))

            if labels is None:
                labels = ['Biological Model']

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'{labels[0]} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{title} - ROC Curve", fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            filepath = os.path.join(self.working_dir, "biological_roc_curve.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating ROC curve: {e}")
            plt.close()
            return None

    def plot_phase_portrait(self, x_data, y_data, trajectories=None, title="System Phase Portrait"):
        """Create phase portrait for dynamical systems in theoretical biology"""
        try:
            plt.figure(figsize=(10, 8))

            # Plot state space points
            plt.scatter(x_data, y_data, alpha=0.6, s=1, color='blue', label='State Space')

            # Plot trajectories if provided
            if trajectories is not None:
                for traj in trajectories:
                    if len(traj) > 1:
                        traj_x, traj_y = zip(*traj)
                        plt.plot(traj_x, traj_y, alpha=0.8, linewidth=1.5, color='red')

            plt.xlabel('Variable 1 (x)')
            plt.ylabel('Variable 2 (y)')
            plt.title(f"{title}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            filepath = os.path.join(self.working_dir, "phase_portrait.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating phase portrait: {e}")
            plt.close()
            return None

    def plot_bifurcation_diagram(self, parameter_values, equilibrium_values, stability_data=None,
                               title="Bifurcation Analysis"):
        """Plot bifurcation diagram showing how equilibria change with parameters"""
        try:
            plt.figure(figsize=(10, 8))

            # Plot bifurcation points
            plt.scatter(parameter_values, equilibrium_values, s=20, alpha=0.7, color='blue')

            # Color by stability if provided
            if stability_data is not None:
                stable_mask = stability_data == 'stable'
                unstable_mask = stability_data == 'unstable'
                plt.scatter(parameter_values[stable_mask], equilibrium_values[stable_mask],
                          s=30, color='green', alpha=0.8, label='Stable')
                plt.scatter(parameter_values[unstable_mask], equilibrium_values[unstable_mask],
                          s=30, color='red', alpha=0.8, label='Unstable')

            plt.xlabel('Parameter Value')
            plt.ylabel('Equilibrium Value')
            plt.title(f"{title}", fontsize=14)
            plt.grid(True, alpha=0.3)
            if stability_data is not None:
                plt.legend()

            filepath = os.path.join(self.working_dir, "bifurcation_diagram.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating bifurcation diagram: {e}")
            plt.close()
            return None

    def plot_eigenvalue_spectrum(self, eigenvalues_real, eigenvalues_imag=None,
                               title="Eigenvalue Spectrum"):
        """Plot eigenvalues in complex plane to assess system stability"""
        try:
            plt.figure(figsize=(8, 8))

            if eigenvalues_imag is None:
                # Real eigenvalues only
                plt.scatter(eigenvalues_real, [0]*len(eigenvalues_real),
                          s=50, alpha=0.8, color='blue', marker='x')
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                plt.xlabel('Real Part')
                plt.ylabel('Imaginary Part')
            else:
                # Complex eigenvalues
                plt.scatter(eigenvalues_real, eigenvalues_imag, s=50, alpha=0.8,
                          color='blue', marker='x')

                # Stability boundaries
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

                # Unit circle for reference
                theta = np.linspace(0, 2*np.pi, 100)
                plt.plot(np.cos(theta), np.sin(theta), color='gray', alpha=0.3,
                        linestyle=':', label='Unit Circle')

                plt.xlabel('Real Part')
                plt.ylabel('Imaginary Part')

            plt.title(f"{title}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')

            filepath = os.path.join(self.working_dir, "eigenvalue_spectrum.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating eigenvalue spectrum plot: {e}")
            plt.close()
            return None

    def plot_agent_based_evolution(self, time_steps, population_data, trait_frequencies=None,
                                 title="Agent-Based Model Evolution"):
        """Plot evolution in agent-based biological models"""
        try:
            plt.figure(figsize=(12, 8))

            # Plot population dynamics if multiple populations
            if isinstance(population_data, dict):
                for pop_name, pop_size in population_data.items():
                    plt.plot(time_steps, pop_size, label=f'{pop_name} Population', linewidth=2)
            else:
                plt.plot(time_steps, population_data, label='Population Size', linewidth=2)

            plt.xlabel('Time Step')
            plt.ylabel('Population Size')
            plt.title(f"{title}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # If trait frequencies provided, create subplot
            if trait_frequencies is not None:
                plt.figure(figsize=(12, 6))

                # Main plot - population
                plt.subplot(1, 2, 1)
                if isinstance(population_data, dict):
                    for pop_name, pop_size in population_data.items():
                        plt.plot(time_steps, pop_size, label=f'{pop_name} Population', linewidth=2)
                else:
                    plt.plot(time_steps, population_data, label='Population Size', linewidth=2)
                plt.xlabel('Time Step')
                plt.ylabel('Population Size')
                plt.title('Population Dynamics')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Trait frequency plot
                plt.subplot(1, 2, 2)
                for trait_name, frequency in trait_frequencies.items():
                    plt.plot(time_steps, frequency, label=f'{trait_name}', linewidth=2)
                plt.xlabel('Time Step')
                plt.ylabel('Trait Frequency')
                plt.title('Trait Evolution')
                plt.grid(True, alpha=0.3)
                plt.legend()

                filepath = os.path.join(self.working_dir, "agent_based_evolution_combined.png")
            else:
                filepath = os.path.join(self.working_dir, "agent_based_evolution.png")

            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error creating agent-based evolution plot: {e}")
            plt.close()
            return None

    def plot_fitness_landscape(self, parameter_grid, fitness_values, optimal_points=None,
                elif mean1 == 0:
                    fold_change = float('inf')
                    log2_fc = float('inf')
                elif mean2 == 0:
                    fold_change = 0
                    log2_fc = float('-inf')
                else:
                    fold_change = mean2 / mean1
                    log2_fc = np.log2(fold_change)

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(group1_data[i], group2_data[i])

                results.append({
                    'gene_index': i,
                    'fold_change': fold_change,
                    'log2_fold_change': log2_fc,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

            return results

        except Exception as e:
            print(f"Error in differential expression analysis: {e}")
            return None


def aggregate_biological_plots(base_folder, model="gpt-4o-2024-11-20"):
    """
    Aggregate biological plots from experiment results.
    Enhanced version of aggregate_plots for biological data.
    """
    # This would integrate with the existing aggregate_plots functionality
    # but with biological-specific enhancements
    print("Aggregating biological experiment plots...")

    # Import and use existing functionality as base
    try:
        from ai_scientist.perform_plotting import aggregate_plots as base_aggregate
        return base_aggregate(base_folder, model)
    except ImportError:
        print("Could not import base plotting functionality")
        return None


# Convenience function for easy biological plotting
def create_biological_plots(experiment_data_path="./working/experiment_data.npy",
                          plot_types=None):
    """
    Create standard biological plots from experiment data.

    Args:
        experiment_data_path: Path to numpy data file
        plot_types: List of plot types to create ('heatmap', 'network', 'volcano', etc.)
    """
    if plot_types is None:
        plot_types = ['heatmap', 'volcano', 'roc']

    try:
        experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
        plotter = BiologicalPlotter()

        plots_created = []

        for plot_type in plot_types:
            if plot_type == 'heatmap' and 'expression_data' in experiment_data:
                plot_path = plotter.plot_gene_expression_heatmap(
                    experiment_data['expression_data']
                )
                if plot_path:
                    plots_created.append(plot_path)

            elif plot_type == 'volcano' and 'fold_changes' in experiment_data and 'p_values' in experiment_data:
                plot_path = plotter.plot_volcano_plot(
                    experiment_data['fold_changes'],
                    experiment_data['p_values']
                )
                if plot_path:
                    plots_created.append(plot_path)

            elif plot_type == 'roc' and 'y_true' in experiment_data and 'y_scores' in experiment_data:
                plot_path = plotter.plot_roc_curve_biological(
                    experiment_data['y_true'],
                    experiment_data['y_scores']
                )
                if plot_path:
                    plots_created.append(plot_path)

        return plots_created

    except Exception as e:
        print(f"Error creating biological plots: {e}")
        return []
