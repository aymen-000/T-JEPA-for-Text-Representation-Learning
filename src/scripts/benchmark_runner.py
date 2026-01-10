"""
Comprehensive Benchmark Runner for Text-JEPA

Runs all downstream tasks and generates a summary report.
Perfect for research paper results tables.
"""

import subprocess
import argparse
import os
from datetime import datetime
import json


TASKS = {
    "text_classification": {
        "script": "src/evaluation/finetune_text_classification.py",
        "name": "AG News (Text Classification)",
        "epochs": 5,
        "batch_size": 64,
    },
    "sentiment": {
        "script": "src/evaluation/finetune_sentiment.py",
        "name": "SST-2 (Sentiment Analysis)",
        "epochs": 10,
        "batch_size": 32,
    },
    "paraphrase_mrpc": {
        "script": "src/evaluation/finetune_paraphrase.py",
        "name": "MRPC (Paraphrase Detection)",
        "epochs": 15,
        "batch_size": 32,
        "extra_args": ["--dataset", "mrpc"],
    },
    "paraphrase_qqp": {
        "script": "src/evaluation/finetune_paraphrase.py",
        "name": "QQP (Paraphrase Detection)",
        "epochs": 15,
        "batch_size": 32,
        "extra_args": ["--dataset", "qqp"],
    },
}


def run_task(task_name, task_config, checkpoint, config, model_name, 
             lr, encoder_lr, device, output_dir):
    """
    Run a single task and return the results.
    """
    print("\n" + "=" * 70)
    print(f"RUNNING: {task_config['name']}")
    print("=" * 70 + "\n")
    
    # Check if script exists
    script_path = task_config["script"]
    if not os.path.exists(script_path):
        return {
            "status": "failed",
            "error": f"Script not found: {script_path}",
            "output": "",
        }
    
    cmd = [
        "python", script_path,
        "--checkpoint", checkpoint,
        "--config", config,
        "--epochs", str(task_config["epochs"]),
        "--batch_size", str(task_config["batch_size"]),
        "--lr", str(lr),
        "--encoder_lr", str(encoder_lr),
        "--device", device,
        "--output_dir", os.path.join(output_dir, task_name),
    ]
    
    if model_name:
        cmd.extend(["--model_name", model_name])
    
    # Add any extra arguments
    if "extra_args" in task_config:
        cmd.extend(task_config["extra_args"])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        # Parse output for accuracy/F1
        output = result.stdout
        print(output)  # Print output in real-time
        
        # Try to extract the final metric
        metric = None
        if "BEST FINE-TUNING ACCURACY:" in output:
            metric = output.split("BEST FINE-TUNING ACCURACY:")[1].split("%")[0].strip()
            metric = float(metric)
        elif "BEST SENTIMENT ACCURACY:" in output:
            metric = output.split("BEST SENTIMENT ACCURACY:")[1].split("%")[0].strip()
            metric = float(metric)
        elif "BEST F1 SCORE:" in output:
            metric = output.split("BEST F1 SCORE:")[1].strip()
            metric = float(metric) * 100  # Convert to percentage
        
        return {
            "status": "success",
            "metric": metric,
            "output": output,
        }
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return {
            "status": "failed",
            "error": str(e),
            "output": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else "",
        }
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "output": "",
        }


def generate_summary_report(results, output_dir):
    """
    Generate a comprehensive summary report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.txt")
    json_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    latex_path = os.path.join(output_dir, f"latex_table_{timestamp}.txt")
    
    # Text report
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT-JEPA COMPREHENSIVE BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 70 + "\n")
        
        for task_name, result in results.items():
            task_display = TASKS[task_name]["name"]
            f.write(f"\n{task_display}:\n")
            
            if result["status"] == "success":
                if result["metric"] is not None:
                    f.write(f"  ✓ Score: {result['metric']:.2f}%\n")
                else:
                    f.write(f"  ✓ Completed (metric not parsed)\n")
            else:
                f.write(f"  ✗ Failed: {result.get('error', 'Unknown error')}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED BREAKDOWN\n")
        f.write("=" * 70 + "\n\n")
        
        # Categorize by task type
        f.write("SINGLE-SENTENCE CLASSIFICATION:\n")
        f.write("-" * 70 + "\n")
        for task in ["text_classification", "sentiment"]:
            if task in results and results[task]["status"] == "success":
                task_display = TASKS[task]["name"]
                metric = results[task]["metric"]
                f.write(f"  {task_display}: {metric:.2f}%\n")
        
        f.write("\nSENTENCE-PAIR TASKS:\n")
        f.write("-" * 70 + "\n")
        for task in [ "paraphrase_mrpc", "paraphrase_qqp"]:
            if task in results and results[task]["status"] == "success":
                task_display = TASKS[task]["name"]
                metric = results[task]["metric"]
                f.write(f"  {task_display}: {metric:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Calculate average
        successful_metrics = [
            r["metric"] for r in results.values() 
            if r["status"] == "success" and r["metric"] is not None
        ]
        
        if successful_metrics:
            avg = sum(successful_metrics) / len(successful_metrics)
            f.write(f"Average Score across {len(successful_metrics)} tasks: {avg:.2f}%\n")
            f.write(f"Best Score: {max(successful_metrics):.2f}%\n")
            f.write(f"Worst Score: {min(successful_metrics):.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    # JSON report
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # LaTeX table
    with open(latex_path, "w") as f:
        f.write("% LaTeX table for research paper\n")
        f.write("% Copy this into your paper's results section\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Text-JEPA Performance on Downstream Tasks}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Task} & \\textbf{Score (\\%)} \\\\\n")
        f.write("\\hline\n")
        
        for task_name, result in results.items():
            if result["status"] == "success" and result["metric"] is not None:
                task_display = TASKS[task_name]["name"]
                metric = result["metric"]
                f.write(f"{task_display} & {metric:.2f} \\\\\n")
        
        f.write("\\hline\n")
        
        if successful_metrics:
            avg = sum(successful_metrics) / len(successful_metrics)
            f.write(f"\\textbf{{Average}} & \\textbf{{{avg:.2f}}} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\end{table}\n")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\n✓ Summary report: {report_path}")
    print(f"✓ JSON results: {json_path}")
    print(f"✓ LaTeX table: {latex_path}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive Text-JEPA benchmark"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--script_dir", type=str, default="src/evaluation",
                        help="Directory containing evaluation scripts")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (bert-base-uncased, gpt2, etc.)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for classifier head")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Learning rate for encoder")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark",
                        help="Output directory for all results")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to run (default: all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update script paths with script_dir
    for task_name in TASKS:
        script_basename = os.path.basename(TASKS[task_name]["script"])
        TASKS[task_name]["script"] = os.path.join(args.script_dir, script_basename)
    
    # Determine which tasks to run
    tasks_to_run = args.tasks if args.tasks else list(TASKS.keys())
    
    print("\n" + "=" * 70)
    print("TEXT-JEPA COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Script directory: {args.script_dir}")
    print(f"Tasks to run: {', '.join(tasks_to_run)}")
    print(f"Output directory: {args.output_dir}")
    print("\n" + "=" * 70)
    
    # Verify scripts exist
    print("\nVerifying scripts...")
    missing_scripts = []
    for task_name in tasks_to_run:
        if task_name in TASKS:
            script_path = TASKS[task_name]["script"]
            if os.path.exists(script_path):
                print(f"  ✓ {script_path}")
            else:
                print(f"  ✗ {script_path} (NOT FOUND)")
                missing_scripts.append(script_path)
    
    if missing_scripts:
        print(f"\nERROR: {len(missing_scripts)} script(s) not found!")
        print("Please check the --script_dir argument or script paths.")
        return
    
    print("\n" + "=" * 70)
    
    # Run all tasks
    results = {}
    
    for task_name in tasks_to_run:
        if task_name not in TASKS:
            print(f"\nWarning: Unknown task '{task_name}', skipping...")
            continue
        
        task_config = TASKS[task_name]
        
        result = run_task(
            task_name=task_name,
            task_config=task_config,
            checkpoint=args.checkpoint,
            config=args.config,
            model_name=args.model_name,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            device=args.device,
            output_dir=args.output_dir,
        )
        
        results[task_name] = result
    
    # Generate summary report
    generate_summary_report(results, args.output_dir)


if __name__ == "__main__":
    main()