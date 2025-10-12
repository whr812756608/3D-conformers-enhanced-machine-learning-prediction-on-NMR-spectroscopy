"""
1H_NMR_running_pipeline.py - Master pipeline controller for ¹H NMR prediction
Orchestrates pretraining, training, and testing with centralized config

IMPORTANT: H NMR uses the SAME pretrained model as C NMR!
The pretraining step is shared between both nuclei.

Usage:
    # Interactive mode (just click Run or execute without args)
    python 1H_NMR_running_pipeline.py

    # Command-line mode
    python 1H_NMR_running_pipeline.py --all              # Run all stages
    python 1H_NMR_running_pipeline.py --train --test     # Run training and testing only
    python 1H_NMR_running_pipeline.py --test             # Run testing only
    python 1H_NMR_running_pipeline.py --pretrain         # Run pretraining only (shared with C NMR)
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess
import H_NMR_config as config


class HNMRPipelineRunner:
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
        print("¹H NMR EXPERIMENT CONFIGURATION")
        print("=" * 80)
        print(f"  Seeds: {config.SEEDS}")
        print(f"  Pretrained Init: {'ON' if config.IF_PRETRAIN else 'OFF'}")
        print(f"  Epochs: {config.EPOCHS}")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Learning Rate: {config.LR}")
        print(f"  Weight Decay: {config.WEIGHT_DECAY}")
        print(f"  Early Patience: {config.EARLY_PATIENCE}")
        print(f"  Top N Atoms: {config.TOP_N_ATOMS}")
        print("=" * 80)
        print("NOTE: H NMR uses the SAME pretrained model as C NMR!")
        print(f"      Shared pretrained path: {config.PRETRAINED_PATH}")
        print("=" * 80 + "\n")

    def run_pretrain(self):
        """
        Run pretraining stage.

        IMPORTANT: This is the SAME pretraining used for C NMR!
        The 3D encoder learns universal geometric features that transfer to both nuclei.
        """
        stage_start = time.time()
        self.print_header("PRETRAINING (SHARED WITH C NMR)")

        # Same pretrain script as C NMR!
        pretrain_script = self.repo_root / "GEOM_pretrain" / "GEOM_3D_pretraining.py"

        if not pretrain_script.exists():
            print(f"Warning: Pretrain script not found at {pretrain_script}")
            print("Skipping pretraining stage.")
            self.results['pretrain'] = 'skipped'
            return True

        try:
            # Check if pretrained model already exists
            if config.PRETRAINED_PATH.exists():
                print(f"✓ Found existing pretrained model at: {config.PRETRAINED_PATH}")
                print("NOTE: This pretrained model is SHARED between 13C and 1H NMR.")
                print("      If you ran 13C NMR pretraining, this is the SAME model.")
                response = input("\nUse existing pretrained model? (Y/n): ").strip().lower()

                if response == '' or response == 'y':
                    print("✓ Using existing pretrained model (recommended).")
                    self.results['pretrain'] = 'existing'
                    elapsed = time.time() - stage_start
                    self.print_footer("PRETRAINING", elapsed, True)
                    return True
                else:
                    confirm = input("Are you sure you want to retrain and OVERWRITE? (y/N): ").strip().lower()
                    if confirm != 'y':
                        print("✓ Using existing pretrained model.")
                        self.results['pretrain'] = 'existing'
                        elapsed = time.time() - stage_start
                        self.print_footer("PRETRAINING", elapsed, True)
                        return True

            print(f"Running: python {pretrain_script}")
            print("This pretrained model will be used by BOTH ¹³C and ¹H NMR!\n")
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
        """Run ¹H NMR training stage."""
        stage_start = time.time()
        self.print_header("¹H NMR TRAINING")

        train_script = self.repo_root / "GEOM_1H_NMR" / "1H_NMR_training.py"

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
            self.print_footer("1H NMR TRAINING", elapsed, True)
            self.results['train'] = 'success'
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - stage_start
            print(f"\nError during training: {e}")
            self.print_footer("1H NMR TRAINING", elapsed, False)
            self.results['train'] = 'failed'
            return False
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"\nUnexpected error during training: {e}")
            self.print_footer("1H NMR TRAINING", elapsed, False)
            self.results['train'] = 'failed'
            return False

    def run_test(self):
        """Run ¹H NMR testing stage."""
        stage_start = time.time()
        self.print_header("¹H NMR TESTING")

        test_script = self.repo_root / "GEOM_1H_NMR" / "1H_NMR_testing.py"

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
            self.print_footer("1H NMR TESTING", elapsed, True)
            self.results['test'] = 'success'
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - stage_start
            print(f"\nError during testing: {e}")
            self.print_footer("¹H NMR TESTING", elapsed, False)
            self.results['test'] = 'failed'
            return False
        except Exception as e:
            elapsed = time.time() - stage_start
            print(f"\nUnexpected error during testing: {e}")
            self.print_footer("1H NMR TESTING", elapsed, False)
            self.results['test'] = 'failed'
            return False

    def print_final_summary(self):
        """Print final pipeline summary."""
        print("\n" + "=" * 80)
        print("=" * 24 + " 1H NMR PIPELINE SUMMARY " + "=" * 24)
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
        print(f"  Pretrained: {'ON' if config.IF_PRETRAIN else 'OFF'}")
        print(f"  Epochs: {config.EPOCHS}")

        print(f"\nOutput Locations:")
        print(f"  Models: {config.DIR_MODELS}")
        print(f"  Pretrained: {config.DIR_PRETRAIN}")

        print(f"\nIMPORTANT:")
        print(f"  ✓ Pretrained model is SHARED with 13C NMR")
        print(f"  ✓ Only train/test stages are nucleus-specific")

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
        Run the complete ¹H NMR pipeline or selected stages.

        Args:
            run_pretrain: Whether to run pretraining (shared with C NMR)
            run_train: Whether to run ¹H NMR training
            run_test: Whether to run ¹H NMR testing
        """
        self.start_time = time.time()

        print("\n" + "=" * 80)
        print("=" * 20 + " ¹H NMR PREDICTION PIPELINE " + "=" * 18)
        print("=" * 80)
        print(f"Repository: {self.repo_root}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Show configuration
        self.print_config()

        # Show what will be executed
        stages = []
        if run_pretrain: stages.append("PRETRAIN (shared)")
        if run_train: stages.append("TRAIN (¹H)")
        if run_test: stages.append("TEST (¹H)")

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
    print("=" * 20 + " ¹H NMR PREDICTION PIPELINE " + "=" * 18)
    print("=" * 20 + "     INTERACTIVE MODE        " + "=" * 18)
    print("=" * 80)

    print("\nCurrent configuration (edit H_NMR_config.py to change):")
    print(f"  Seeds: {config.SEEDS}")
    print(f"  Pretrained Init: {'ON' if config.IF_PRETRAIN else 'OFF'}")
    print(f"  Epochs: {config.EPOCHS}")

    print("Pretraining is SHARED between 13C and 1H NMR")
    print("If you already ran ¹³C pretraining, you can skip the pretrain step.")

    print("\n" + "-" * 80)
    print("Select stages to run:")
    print("-" * 80)

    # Ask for each stage
    print("\n1. Run PRETRAINING (shared with ¹³C)? (y/N): ", end='')
    run_pretrain = input().strip().lower() == 'y'

    print("2. Run ¹H NMR TRAINING? (y/N): ", end='')
    run_train = input().strip().lower() == 'y'

    print("3. Run ¹H NMR TESTING? (y/N): ", end='')
    run_test = input().strip().lower() == 'y'

    # Check if at least one stage selected
    if not (run_pretrain or run_train or run_test):
        print("\n" + "=" * 80)
        print("No stages selected. Exiting.")
        print("=" * 80 + "\n")
        return False

    # Confirm
    stages = []
    if run_pretrain: stages.append("Pretrain (shared)")
    if run_train: stages.append("Train (¹H)")
    if run_test: stages.append("Test (¹H)")

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
    runner = HNMRPipelineRunner()
    success = runner.run_pipeline(
        run_pretrain=run_pretrain,
        run_train=run_train,
        run_test=run_test
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description='¹H NMR Prediction Pipeline Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments)
  python 1H_NMR_running_pipeline.py

  # Command-line mode
  python 1H_NMR_running_pipeline.py --all              # Run all stages
  python 1H_NMR_running_pipeline.py --train --test     # Run training and testing (skip pretrain)
  python 1H_NMR_running_pipeline.py --test             # Run testing only
  python 1H_NMR_running_pipeline.py --pretrain         # Run pretraining only (shared with ¹³C)

Configuration:
  Edit H_NMR_config.py to change settings (seeds, epochs, etc.)
  All scripts automatically use H_NMR_config.py settings.

IMPORTANT:
  - Pretraining is SHARED between ¹³C and ¹H NMR
  - If you already ran ¹³C pretraining, you can skip the --pretrain step
  - The same 3D_pretrained.pt is used for both nuclei
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all stages (pretrain, train, test)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run pretraining stage (shared with 13C NMR)')
    parser.add_argument('--train', action='store_true',
                        help='Run ¹H NMR training stage')
    parser.add_argument('--test', action='store_true',
                        help='Run ¹H NMR testing stage')

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
    runner = HNMRPipelineRunner()
    success = runner.run_pipeline(
        run_pretrain=args.pretrain,
        run_train=args.train,
        run_test=args.test
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()