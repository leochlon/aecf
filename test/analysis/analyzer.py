"""
Results analysis and comparison functionality.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class ResultsAnalyzer:
    """Analyzes and compares ablation results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def analyze_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze and compare ablation results."""
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        # Extract key metrics including test_ece and mAP
        analysis_df = pd.DataFrame({
            'Ablation': df['ablation_name'],
            'Test_Loss': df['test_metrics'].apply(lambda x: x.get('test_loss', float('nan'))),
            'Test_Accuracy': df['test_metrics'].apply(lambda x: x.get('test_acc', float('nan'))),
            'Test_ECE': df['test_metrics'].apply(lambda x: x.get('ece', float('nan'))),
            'Test_MAP': df['test_metrics'].apply(lambda x: x.get('mAP', float('nan'))),
            'Best_Val_Loss': df['best_val_loss'],
            'Epochs': df['epochs_trained'],
            'Modalities': df['config'].apply(lambda x: len(x.get('modalities', []))),
            'Gate_Enabled': df['config'].apply(lambda x: not x.get('gate_disabled', False)),
            'Entropy_Reg': df['config'].apply(lambda x: x.get('entropy_reg', True)),
            'Curriculum': df['config'].apply(lambda x: x.get('curriculum_mask', True))
        })
        
        return analysis_df
    
    def save_results(self, results: List[Dict[str, Any]], analysis_df: pd.DataFrame):
        """Save results and analysis."""
        # Save raw results
        with open(self.output_dir / 'raw_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save analysis table
        analysis_df.to_csv(self.output_dir / 'analysis_summary.csv', index=False)
        
        # Save formatted report
        self._generate_report(analysis_df)
    
    def _generate_report(self, df: pd.DataFrame):
        """Generate formatted analysis report."""
        report = []
        report.append("# AECF Ablation Study Results\n")
        
        # Summary table
        report.append("## Results Summary")
        report.append(df.to_string(index=False))
        report.append("\n")
        
        # Component analysis
        if len(df) > 1:
            baseline = df[df['Ablation'] == 'full']
            if not baseline.empty:
                baseline_acc = baseline['Test_Accuracy'].iloc[0]
                report.append("## Component Analysis")
                
                for _, row in df.iterrows():
                    if row['Ablation'] != 'full':
                        diff = row['Test_Accuracy'] - baseline_acc
                        report.append(f"- {row['Ablation']}: {diff:+.4f} vs baseline")
                
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
