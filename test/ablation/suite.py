"""
Main ablation suite orchestrator.
"""
import logging
import multiprocessing as mp
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from .config import AblationConfig
from .experiment import AblationExperiment
from data.manager import COCODataManager
from analysis.analyzer import ResultsAnalyzer


class AblationSuite:
    """Main ablation suite orchestrator with integrated data management."""
    
    # Predefined ablation configurations with rigorous validation
    STANDARD_ABLATIONS = {
        "full": AblationConfig(name="full"),
        "no_gate": AblationConfig(name="no_gate", gate_disabled=True),
        "no_entropy": AblationConfig(name="no_entropy", entropy_reg=False),
        "no_curmask": AblationConfig(name="no_curmask", curriculum_mask=False),
        "img_only": AblationConfig(name="img_only", modalities=["image"]),
        "txt_only": AblationConfig(name="txt_only", modalities=["text"])
    }
    
    def __init__(self, 
                 data_root: Path = Path("/content/coco2014"),
                 cache_dir: Path = Path("./cache"),
                 output_dir: Path = Path("./ablation_results"),
                 parallel: bool = False):
        
        self.data_manager = COCODataManager(data_root, cache_dir)
        self.output_dir = output_dir
        self.parallel = parallel
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_main_logging()
        
        # Validate system requirements
        self._validate_system_requirements()
        
    def _setup_main_logging(self) -> logging.Logger:
        """Setup main suite logging."""
        logger = logging.getLogger("ablation_suite")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.output_dir / "suite.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _validate_system_requirements(self):
        """Validate system has required capabilities."""
        self.logger.info("🔍 Validating system requirements...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            self.logger.warning("⚠️ CUDA not available, using CPU (will be slow)")
        
        # Check memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 4:
                self.logger.warning(f"⚠️ Low GPU memory: {gpu_memory:.1f}GB")
            else:
                self.logger.info(f"✅ GPU memory: {gpu_memory:.1f}GB")
        
        # Check CPU cores for data loading
        cpu_count = mp.cpu_count()
        self.logger.info(f"✅ CPU cores: {cpu_count}")
        
        self.logger.info("✅ System requirements validated")
    
    def run_ablations(self, 
                     ablation_names: Optional[List[str]] = None,
                     custom_configs: Optional[Dict[str, AblationConfig]] = None) -> pd.DataFrame:
        """Run ablation experiments with comprehensive validation."""
        
        # Determine which ablations to run
        if custom_configs:
            configs = custom_configs
        else:
            if ablation_names is None:
                ablation_names = list(self.STANDARD_ABLATIONS.keys())
            configs = {name: self.STANDARD_ABLATIONS[name] for name in ablation_names}
        
        # Update output directories for all configs
        for config in configs.values():
            config.output_dir = self.output_dir
        
        self.logger.info(f"Running {len(configs)} ablations: {list(configs.keys())}")
        
        # CRITICAL: Ensure data is ready with comprehensive validation
        print("🔍 Preparing COCO data with validation...")
        if not self.data_manager.ensure_data_ready():
            raise RuntimeError("Data preparation failed - cannot proceed with ablations")
        
        # Get data info for logging
        info = self.data_manager.get_data_info()
        self.logger.info(f"📊 Data ready: {info}")
        
        # Get data loaders (guaranteed to work after ensure_data_ready)
        # Optimize for A100 GPU with 40GB memory and 12 CPU cores
        effective_batch_size = list(configs.values())[0].batch_size
        if torch.cuda.is_available():
            # Maximize batch size for A100 GPU training
            effective_batch_size = max(256, effective_batch_size)  # Increased for A100
            
            # Set optimal number of workers for high-performance system
            num_workers = min(12, mp.cpu_count())  # Use all 12 cores
        else:
            num_workers = min(4, mp.cpu_count())
        
        train_loader, val_loader, test_loader = self.data_manager.get_loaders(
            batch_size=effective_batch_size,
            num_workers=num_workers
        )
        
        # Log data loader information
        self._log_data_loader_info(train_loader, val_loader, test_loader)
        
        # Run experiments
        if self.parallel and len(configs) > 1:
            results = self._run_parallel(configs, train_loader, val_loader, test_loader)
        else:
            results = self._run_sequential(configs, train_loader, val_loader, test_loader)
        
        # Analyze results
        analyzer = ResultsAnalyzer(self.output_dir)
        analysis_df = analyzer.analyze_results(results)
        analyzer.save_results(results, analysis_df)
        
        self.logger.info("✅ Ablation suite completed successfully!")
        return analysis_df
    
    def _log_data_loader_info(self, train_loader, val_loader, test_loader):
        """Log information about data loaders for debugging."""
        try:
            train_batch = next(iter(train_loader))
            
            self.logger.info(f"📊 Data loader info:")
            self.logger.info(f"  - Train batches: {len(train_loader)}")
            self.logger.info(f"  - Val batches: {len(val_loader)}")
            self.logger.info(f"  - Test batches: {len(test_loader)}")
            self.logger.info(f"  - Batch keys: {list(train_batch.keys())}")
            
            for key, tensor in train_batch.items():
                if isinstance(tensor, torch.Tensor):
                    self.logger.info(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log data loader info: {e}")
    
    def _run_sequential(self, configs, train_loader, val_loader, test_loader) -> List[Dict]:
        """Run ablations sequentially with proper cleanup."""
        results = []
        total_ablations = len(configs)
        
        for i, (name, config) in enumerate(configs.items(), 1):
            print(f"\n{'='*60}")
            print(f"🔬 ABLATION {i}/{total_ablations}: {name.upper()}")
            print(f"{'='*60}")
            self.logger.info(f"🚀 Running ablation {i}/{total_ablations}: {name}")
            
            try:
                experiment = AblationExperiment(config)
                result = experiment.run(train_loader, val_loader, test_loader)
                results.append(result)
                
                print(f"✅ COMPLETED: {name} (epochs: {result.get('epochs_trained', 'N/A')})")
                self.logger.info(f"✅ Completed ablation: {name}")
                
            except Exception as e:
                print(f"❌ FAILED: {name} - {str(e)}")
                self.logger.error(f"❌ Failed ablation {name}: {e}")
                # Continue with other ablations
                results.append({
                    'ablation_name': name,
                    'config': config.__dict__,
                    'error': str(e),
                    'test_metrics': {},
                    'best_val_loss': float('inf'),
                    'epochs_trained': 0,
                    'model_path': None
                })
            
            finally:
                # Aggressive GPU memory cleanup between experiments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all operations to complete
                    
                # Clean up any lingering references
                if 'experiment' in locals():
                    del experiment
                    
                # Force garbage collection
                import gc
                gc.collect()
                    
        return results
    
    def _run_parallel(self, configs, train_loader, val_loader, test_loader) -> List[Dict]:
        """Run ablations in parallel (experimental - may have GPU memory issues)."""
        self.logger.warning("⚠️ Parallel execution is experimental and may cause GPU memory issues")
        
        def run_single(name_config):
            name, config = name_config
            try:
                experiment = AblationExperiment(config)
                return experiment.run(train_loader, val_loader, test_loader)
            except Exception as e:
                return {
                    'ablation_name': name,
                    'config': config.__dict__,
                    'error': str(e),
                    'test_metrics': {},
                    'best_val_loss': float('inf'),
                    'epochs_trained': 0,
                    'model_path': None
                }
        
        # Limit parallelism to avoid GPU memory issues
        max_workers = min(2, len(configs))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single, configs.items()))
        
        return results
