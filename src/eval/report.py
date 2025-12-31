"""
Genera report markdown e plot
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_file: str) -> dict:
    """Carica risultati da JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def generate_markdown_report(results: dict, output_file: Path):
    """Genera report markdown"""
    lines = []
    lines.append("# Report Valutazione Compressione Latente\n")
    lines.append(f"Esperimento: {results['experiment']}\n")
    
    # Tabella risultati
    lines.append("## Risultati\n")
    lines.append("| Metodo | Exact Match | Token F1 | Latency (s) |\n")
    lines.append("|--------|-------------|----------|-------------|\n")
    
    for result in results["results"]:
        method = result.get("method", "unknown")
        em = result.get("exact_match", 0.0)
        f1 = result.get("token_f1", 0.0)
        latency = result.get("avg_latency", 0.0)
        lines.append(f"| {method} | {em:.4f} | {f1:.4f} | {latency:.4f} |\n")
    
    # VRAM
    lines.append("\n## VRAM Usage\n")
    vram = results.get("vram", {})
    lines.append(f"- Allocated: {vram.get('allocated', 0):.2f} GB\n")
    lines.append(f"- Peak: {vram.get('max_allocated', 0):.2f} GB\n")
    
    # Conclusioni
    lines.append("\n## Conclusioni\n")
    lines.append("Analisi dei risultati:\n")
    
    # Trova best F1
    best_f1 = max(r.get("token_f1", 0) for r in results["results"])
    best_method = next(r["method"] for r in results["results"] if r.get("token_f1", 0) == best_f1)
    lines.append(f"- Miglior F1: {best_method} ({best_f1:.4f})\n")
    
    # Trova fastest
    fastest = min(r.get("avg_latency", float('inf')) for r in results["results"])
    fastest_method = next(r["method"] for r in results["results"] if r.get("avg_latency", float('inf')) == fastest)
    lines.append(f"- Più veloce: {fastest_method} ({fastest:.4f}s)\n")
    
    # Scrivi file
    output_file.write_text("".join(lines))
    print(f"Report markdown salvato in {output_file}")


def generate_plots(results: dict, output_dir: Path):
    """Genera plot"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepara dati
    methods = [r["method"] for r in results["results"]]
    f1_scores = [r.get("token_f1", 0) for r in results["results"]]
    latencies = [r.get("avg_latency", 0) for r in results["results"]]
    
    # Plot 1: F1 vs Latency (Pareto)
    plt.figure(figsize=(10, 6))
    plt.scatter(latencies, f1_scores, s=100, alpha=0.6)
    for i, method in enumerate(methods):
        plt.annotate(method, (latencies[i], f1_scores[i]), fontsize=8)
    plt.xlabel("Latency (s)")
    plt.ylabel("Token F1")
    plt.title("Pareto Plot: Quality vs Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_latency.png", dpi=150)
    plt.close()
    
    # Plot 2: Bar chart F1
    plt.figure(figsize=(12, 6))
    plt.bar(methods, f1_scores)
    plt.xlabel("Metodo")
    plt.ylabel("Token F1")
    plt.title("Confronto Token F1")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=150)
    plt.close()
    
    print(f"Plot salvati in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/report.json")
    parser.add_argument("--output", type=str, default="outputs/report.md")
    args = parser.parse_args()
    
    # Carica risultati
    results = load_results(args.input)
    
    # Genera report
    output_file = Path(args.output)
    generate_markdown_report(results, output_file)
    
    # Genera plot
    plot_dir = output_file.parent / "plots"
    generate_plots(results, plot_dir)
    
    print("Report generato con successo!")


if __name__ == "__main__":
    main()


