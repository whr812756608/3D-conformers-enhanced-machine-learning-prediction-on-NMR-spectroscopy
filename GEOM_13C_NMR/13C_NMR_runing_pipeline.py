"""
run_pipeline.py - Master pipeline controller for NMR prediction
Orchestrates pretraining, training, and testing with centralized config

Usage:
    # Interactive mode (just click Run or execute without args)
    python run_pipeline.py

    # Command-line mode
    python run_pipeline.py --all              # Run all stages
    python run_pipeline.py --train --test     # Run training and testing only
    python run_pipeline.py --test             # Run testing only
    python run_pipeline.py --pretrain         # Run pretraining only
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess
import config


class PipelineRunner:
    def __init__(self):
        self.repo_root = config.REPO_ROOT
        self.start_time = None
        self.results = {
            'pretrain': None,
            'train': None,
            'test': None
        }

    def print_header(self, stage_name):
        """Print a formatted header for each pipeline stage."""
        print("\n" + "=" * 80)
        print(f"{'=' * 30} {stage_name.upper()} {'=' * 30}")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

    def print_footer(self, stage_name, elapsed_time, success):
        """Print a formatted footer after each pipeline stage."""
        status = "COMPLETED" if success else "FAILED"
        print("\n" + "=" * 80)
        print(f"{stage_name.upper()} {status}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        print("=" * 80 + "\n")

    def print_config(self):
        """Print current configuration."""
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 80)
        print(f"  Seeds: {config.SEEDS}")
        print(f"  Normalization: {'ON' if config.USE_NORMALIZATION else 'OFF'}")
        print(f"  Pretrained Init: {'ON' if config.IF_PRETRAIN else 'OFF'}")
        print(f"  Epochs: {config.EPOCHS}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Learning Rate: {config.LR}")
        print(f"  Weight Decay: {config.WEIGHT_DECAY}")
        print(f"  Early Patience: {config.EARLY_PATIENCE}")
        print(f"  Top N Atoms: {config.TOP_N_ATOMS}")
        print("=" * 80 + "\n")

    def run_pretrain(self):
        """Run pretraining stage."""
        stage_start = time.time()
        self.print_header("PRETRAINING")
        pretrain_script = self.repo_root / "GEOM_pretrain" / "pretrain_3D.py"

        if not pretrain_script.exists():
            print(f"Warning: Pretrain script not found at {pretrain_script}")
            print("Skipping pretraining stage.")
            self.results['pretrain'] = 'skipped'
            return True

        try:
            # Check if pretrained model already exists
            if config.PRETRAINED_PATH.exists():
                print(f"Pretrained model already exists at: {config.PRETRAINED_PATH}")
                response = input("Do you want to retrain? (y/N): ").strip().lower()
                if response != 'y':
                    print("Skipping pretraining - using existing model.")
                    self.results['pretrain'] = 'existing'
                    return True

            print(f"Running: python {pretrain_script}")
            result = subprocess.run(
                [sys.executable, str(pretrain_script)],
                cwd=str(self.repo_root),
                check=True
            )

            elapsed = time.time() - stage_start
            self.print_footer("PRETRAINING", elapsed, True)
            self.results['pretrain'] = 'success'
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - stage_start
            print(f"\nError during pretraining: {e}")
            self.print_footer("PRETRAINING", elapsed, False)
            self.results['pretrain'] = 'failed'
            return False
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"\nUnexpected error during pretraining: {e}")
            self.print_footer("PRETRAINING", elapsed, False)
            self.results['pretrain'] = 'failed'
            return False

    def run_train(self):
        """Run training stage."""
        stage_start = time.time()
        self.print_header("TRAINING")

        train_script = self.repo_root / "GEOM_13C_NMR" / "13C_NMR_training.py"

        if not train_script.exists():
            print(f"Error: Training script not found at {train_script}")
            self.results['train'] = 'failed'
            return False

        try:
            print(f"Running: python {train_script}")
            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=str(self.repo_root),
                check=True
            )

            elapsed = time.time() - stage_start
            self.print_footer("TRAINING", elapsed, True)
            self.results['train'] = 'success'
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - stage_start
            print(f"\nError during training: {e}")
            self.print_footer("TRAINING", elapsed, False)
            self.results['train'] = 'failed'
            return False
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"\nUnexpected error during training: {e}")
            self.print_footer("TRAINING", elapsed, False)
            self.results['train'] = 'failed'
            return False

    def run_test(self):
        """Run testing stage."""
        stage_start = time.time()
        self.print_header("TESTING")

        test_script = self.repo_root / "GEOM_13C_NMR" / "13C_NMR_testing.py"

        if not test_script.exists():
            print(f"Error: Testing script not found at {test_script}")
            self.results['test'] = 'failed'
            return False

        try:
            print(f"Running: python {test_script}")
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(self.repo_root),
                check=True
            )

            elapsed = time.time() - stage_start
            self.print_footer("TESTING", elapsed, True)
            self.results['test'] = 'success'
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - stage_start
            print(f"\nError during testing: {e}")
            self.print_footer("TESTING", elapsed, False)
            self.results['test'] = 'failed'
            return False
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"\nUnexpected error during testing: {e}")
            self.print_footer("TESTING", elapsed, False)
            self.results['test'] = 'failed'
            return False

    def print_final_summary(self):
        """Print final pipeline summary."""
        print("\n" + "=" * 80)
        print("=" * 28 + " PIPELINE SUMMARY " + "=" * 28)
        print("=" * 80)

        total_elapsed = time.time() - self.start_time

        print(f"\nTotal Pipeline Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
        print(f"\nStage Results:")

        for stage, result in self.results.items():
            if result is not None:
                status_symbol = {
                    'success': '✓',
                    'failed': '✗',
                    'skipped': '○',
                    'existing': '○'
                }.get(result, '?')
                status_text = result.upper()
                print(f"  {status_symbol} {stage.capitalize():<12}: {status_text}")

        print(f"\nConfiguration Used:")
        print(f"  Seeds: {config.SEEDS}")
        print(f"  Normalization: {'ON' if config.USE_NORMALIZATION else 'OFF'}")
        print(f"  Pretrained: {'ON' if config.IF_PRETRAIN else 'OFF'}")
        print(f"  Epochs: {config.EPOCHS}")

        print(f"\nOutput Locations:")
        print(f"  Models: {config.DIR_MODELS}")
        print(f"  Pretrained: {config.DIR_PRETRAIN}")

        print("\n" + "=" * 80)

        # Check if any stage failed
        failed = any(r == 'failed' for r in self.results.values() if r is not None)
        if failed:
            print("WARNING: Some stages failed. Check logs above for details.")
        else:
            print("All stages completed successfully!")
        print("=" * 80 + "\n")

    def run_pipeline(self, run_pretrain=False, run_train=False, run_test=False):
        """
        Run the complete pipeline or selected stages.

        Args:
            run_pretrain: Whether to run pretraining
            run_train: Whether to run training
            run_test: Whether to run testing
        """
        self.start_time = time.time()

        print("\n" + "=" * 80)
        print("=" * 20 + " NMR PREDICTION PIPELINE " + "=" * 20)
        print("=" * 80)
        print(f"Repository: {self.repo_root}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Show configuration
        self.print_config()

        # Show what will be executed
        stages = []
        if run_pretrain: stages.append("PRETRAIN")
        if run_train: stages.append("TRAIN")
        if run_test: stages.append("TEST")

        print(f"Stages to execute: {' → '.join(stages)}")
        print()

        # Run stages in order
        success = True

        if run_pretrain:
            success = self.run_pretrain()
            if not success:
                print("WARNING: Pretraining failed. Continue anyway? (y/N): ", end='')
                response = input().strip().lower()
                if response != 'y':
                    print("Pipeline aborted.")
                    self.print_final_summary()
                    return False

        if run_train and success:
            success = self.run_train()
            if not success:
                print("WARNING: Training failed. Continue to testing anyway? (y/N): ", end='')
                response = input().strip().lower()
                if response != 'y':
                    print("Pipeline aborted.")
                    self.print_final_summary()
                    return False

        if run_test and success:
            success = self.run_test()

        # Print final summary
        self.print_final_summary()

        return success


def interactive_mode():
    """Interactive mode - ask user which stages to run."""
    print("\n" + "=" * 80)
    print("=" * 22 + " NMR PREDICTION PIPELINE " + "=" * 22)
    print("=" * 22 + "   INTERACTIVE MODE      " + "=" * 22)
    print("=" * 80)

    print("\nCurrent configuration (edit config.py to change):")
    print(f"  Seeds: {config.SEEDS}")
    print(f"  Normalization: {'ON' if config.USE_NORMALIZATION else 'OFF'}")
    print(f"  Pretrained Init: {'ON' if config.IF_PRETRAIN else 'OFF'}")
    print(f"  Epochs: {config.EPOCHS}")

    print("\n" + "-" * 80)
    print("Select stages to run:")
    print("-" * 80)

    # Ask for each stage
    print("\n1. Run PRETRAINING? (y/N): ", end='')
    run_pretrain = input().strip().lower() == 'y'

    print("2. Run TRAINING? (y/N): ", end='')
    run_train = input().strip().lower() == 'y'

    print("3. Run TESTING? (y/N): ", end='')
    run_test = input().strip().lower() == 'y'

    # Check if at least one stage selected
    if not (run_pretrain or run_train or run_test):
        print("\n" + "=" * 80)
        print("No stages selected. Exiting.")
        print("=" * 80 + "\n")
        return False

    # Confirm
    stages = []
    if run_pretrain: stages.append("Pretrain")
    if run_train: stages.append("Train")
    if run_test: stages.append("Test")

    print("\n" + "-" * 80)
    print(f"You selected: {', '.join(stages)}")
    print("Proceed? (Y/n): ", end='')
    confirm = input().strip().lower()

    if confirm == 'n':
        print("\n" + "=" * 80)
        print("Cancelled by user.")
        print("=" * 80 + "\n")
        return False

    # Run the pipeline
    runner = PipelineRunner()
    success = runner.run_pipeline(
        run_pretrain=run_pretrain,
        run_train=run_train,
        run_test=run_test
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description='NMR Prediction Pipeline Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments)
  python run_pipeline.py
  
  # Command-line mode
  python run_pipeline.py --all              # Run all stages
  python run_pipeline.py --train --test     # Run training and testing
  python run_pipeline.py --test             # Run testing only
  python run_pipeline.py --pretrain         # Run pretraining only
  
Configuration:
  Edit config.py to change settings (seeds, normalization, epochs, etc.)
  All scripts automatically use config.py settings.
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all stages (pretrain, train, test)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run pretraining stage')
    parser.add_argument('--train', action='store_true',
                        help='Run training stage')
    parser.add_argument('--test', action='store_true',
                        help='Run testing stage')

    # Check if running with no arguments (interactive mode)
    if len(sys.argv) == 1:
        # Interactive mode
        success = interactive_mode()
        sys.exit(0 if success else 1)

    # Command-line mode
    args = parser.parse_args()

    # If --all is specified, enable all stages
    if args.all:
        args.pretrain = True
        args.train = True
        args.test = True

    # If no stages selected in command-line mode, show help
    if not (args.pretrain or args.train or args.test):
        parser.print_help()
        print("\nNo stages selected. Use --all, or specify --pretrain, --train, and/or --test")
        sys.exit(1)

    # Create pipeline runner and execute
    runner = PipelineRunner()
    success = runner.run_pipeline(
        run_pretrain=args.pretrain,
        run_train=args.train,
        run_test=args.test
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
