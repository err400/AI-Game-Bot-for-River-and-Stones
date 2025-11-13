#!/usr/bin/env python3
"""
Automated grading script for River and Stones student submissions.

Usage:
    python grade_submissions.py <submissions_directory> [options]
    python grade_submissions.py <single_student_directory> --single

This script:
1. Processes each student submission directory
2. Builds C++ submissions if needed
3. Runs simulations against aggressive, neutral, and defensive agents
4. Collects and reports results
"""

import os
import multiprocessing as mp
import sys
import subprocess
import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def log_error(student_dir: Path, message: str):
    """Log error message to logs.txt in student directory."""
    try:
        log_file = student_dir / "logs.txt"
        # Create file if it doesn't exist
        log_file.touch()
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"Warning: Could not write to log file in {student_dir}: {e}")

def run_command(cmd: List[str], cwd: str = None, timeout: int = 300) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)

def copy_grading_files(student_dir: Path) -> bool:
    """Copy gameEngine.py and agent.py to student directory."""
    try:
        script_dir = Path(__file__).parent
        shutil.copy2(script_dir / "gameEngine.py", student_dir / "gameEngine.py")
        shutil.copy2(script_dir / "agent.py", student_dir / "agent.py")
        return True
    except Exception as e:
        error_msg = f"Error copying files to {student_dir}: {e}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False

def handle_dual_submissions(student_dir: Path) -> str:
    """Handle case where student has both Python and C++ files."""
    python_file = student_dir / "student_agent.py"
    cpp_file = student_dir / "student_agent.cpp"
    
    if python_file.exists() and cpp_file.exists():
        # Rename Python file to avoid conflicts
        backup_file = student_dir / "student_agent_cpp.py"
        try:
            shutil.move(str(python_file), str(backup_file))
            log_error(student_dir, f"Renamed student_agent.py to student_agent_cpp.py (C++ submission detected)")
            return "cpp"
        except Exception as e:
            error_msg = f"Failed to rename student_agent.py: {e}"
            log_error(student_dir, error_msg)
            return "error"
    elif python_file.exists():
        return "python"
    elif cpp_file.exists():
        return "cpp"
    else:
        return "none"

def build_cpp_submission(student_dir: Path) -> bool:
    """Build C++ submission using cmake and make."""
    build_dir = student_dir / "build"
    
    # Create build directory
    try:
        build_dir.mkdir(exist_ok=True)
    except Exception as e:
        error_msg = f"Error creating build directory: {e}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False
    
    # Get pybind11 cmake directory
    try:
        pybind11_result = subprocess.run(
            ["python3", "-m", "pybind11", "--cmakedir"],
            capture_output=True, text=True, timeout=30
        )
        if pybind11_result.returncode != 0:
            error_msg = f"Failed to get pybind11 cmake directory: {pybind11_result.stderr}"
            print(error_msg)
            log_error(student_dir, error_msg)
            return False
        pybind11_cmake_dir = pybind11_result.stdout.strip()
    except Exception as e:
        error_msg = f"Error getting pybind11 cmake directory: {e}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False
    
    # Run cmake
    cmake_cmd = [
        "cmake", "..",
        f"-Dpybind11_DIR={pybind11_cmake_dir}",
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++"
    ]
    
    success, stdout, stderr = run_command(cmake_cmd, cwd=str(build_dir))
    if not success:
        error_msg = f"CMake failed for {student_dir.name}: {stderr}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False
    
    # Run make
    success, stdout, stderr = run_command(["make"], cwd=str(build_dir))
    if not success:
        error_msg = f"Make failed for {student_dir.name}: {stderr}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False
    
    return True

def run_simulation(student_dir: Path, circle_strategy: str, square_strategy: str) -> bool:
    """Run simulation and append results to sims.txt."""
    cmd = [
        "python3", "gameEngine.py",
        "--mode", "aivai",
        "--circle", circle_strategy,
        "--square", square_strategy,
        "--nogui"
    ]
    
    success, stdout, stderr = run_command(cmd, cwd=str(student_dir))
    
    if success:
        # Append to sims.txt
        try:
            sims_file = student_dir / "sims.txt"
            # Create file if it doesn't exist
            sims_file.touch()
            with open(sims_file, "a") as f:
                f.write(stdout)
            return True
        except Exception as e:
            error_msg = f"Failed to write simulation output: {e}"
            print(error_msg)
            log_error(student_dir, error_msg)
            return False
    else:
        error_msg = f"Simulation failed for {student_dir.name} ({circle_strategy} vs {square_strategy}): {stderr}"
        print(error_msg)
        log_error(student_dir, error_msg)
        return False

def process_student_submission(student_dir: Path) -> Dict:
    """Process a single student submission."""
    student_name = student_dir.name
    print(f"Processing {student_name}...")
    
    # Clear previous logs and sims, then create empty files
    log_file = student_dir / "logs.txt"
    sims_file = student_dir / "sims.txt"
    if log_file.exists():
        log_file.unlink()
    if sims_file.exists():
        sims_file.unlink()
    
    # Create empty files
    log_file.touch()
    sims_file.touch()
    
    # Handle dual submissions
    submission_type = handle_dual_submissions(student_dir)
    
    if submission_type == "error":
        return {
            "student": student_name,
            "error": "Failed to handle dual submissions"
        }
    elif submission_type == "none":
        error_msg = "No student_agent.py or student_agent.cpp found"
        log_error(student_dir, error_msg)
        return {
            "student": student_name,
            "error": error_msg
        }
    
    # Copy grading files
    if not copy_grading_files(student_dir):
        return {
            "student": student_name,
            "error": "Failed to copy grading files"
        }
    
    # Build C++ if needed
    if submission_type == "cpp":
        if not build_cpp_submission(student_dir):
            return {
                "student": student_name,
                "error": "C++ build failed"
            }
    
    # Run simulations
    strategies = ["aggressive", "neutral", "defensive"]
    student_strategy = "student_cpp" if submission_type == "cpp" else "student"
    successful_runs = 0
    total_runs = 6  # 3 strategies × 2 sides
    
    for strategy in strategies:
        print(f"  Running vs {strategy}...")
        
        # Student as circle vs strategy as square
        if run_simulation(student_dir, student_strategy, strategy):
            successful_runs += 1
        
        # Strategy as circle vs student as square
        if run_simulation(student_dir, strategy, student_strategy):
            successful_runs += 1
    
    # Generate results file
    try:
        results_file = student_dir / "results.txt"
        with open(results_file, "w") as f:
            f.write(f"Student: {student_name}\n")
            f.write(f"Submission Type: {submission_type}\n")
            f.write(f"Successful Simulations: {successful_runs}/{total_runs}\n")
            f.write(f"Results saved in: sims.txt\n")
            if log_file.exists():
                f.write(f"Errors logged in: logs.txt\n")
    except Exception as e:
        error_msg = f"Failed to write results file: {e}"
        print(error_msg)
        log_error(student_dir, error_msg)
    
    return {
        "student": student_name,
        "type": submission_type,
        "successful_runs": successful_runs,
        "total_runs": total_runs
    }

def generate_summary_report(all_results: List[Dict], output_file: str):
    """Generate a summary report."""
    with open(output_file, 'w') as f:
        f.write("River and Stones - Grading Summary\n")
        f.write("=" * 40 + "\n\n")
        
        successful_submissions = [r for r in all_results if "error" not in r]
        failed_submissions = [r for r in all_results if "error" in r]
        
        f.write(f"Total submissions: {len(all_results)}\n")
        f.write(f"Successful submissions: {len(successful_submissions)}\n")
        f.write(f"Failed submissions: {len(failed_submissions)}\n\n")
        
        if successful_submissions:
            f.write("SUCCESSFUL SUBMISSIONS:\n")
            f.write("-" * 25 + "\n")
            for result in successful_submissions:
                f.write(f"{result['student']}: {result['successful_runs']}/{result['total_runs']} runs successful ({result['type']})\n")
        
        if failed_submissions:
            f.write("\nFAILED SUBMISSIONS:\n")
            f.write("-" * 20 + "\n")
            for result in failed_submissions:
                f.write(f"{result['student']}: {result['error']}\n")

def main():
    parser = argparse.ArgumentParser(description="Grade River and Stones student submissions")
    parser.add_argument("submissions_dir", help="Directory containing student submission folders (or single student folder)")
    parser.add_argument("--single", action="store_true", help="Process single student directory")
    parser.add_argument("--summary", "-s", default="grading_summary.txt", help="Summary report file (default: grading_summary.txt)")
    parser.add_argument("--cores", type=int, default=1, help="Number of parallel processes to use (default: 1)")
    
    args = parser.parse_args()
    
    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        print(f"Error: Directory {submissions_dir} does not exist")
        sys.exit(1)
    
    if args.single:
        # Process single student
        if not submissions_dir.is_dir():
            print(f"Error: {submissions_dir} is not a directory")
            sys.exit(1)
        
        print(f"Processing single student: {submissions_dir.name}")
        result = process_student_submission(submissions_dir)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        else:
            print(f"Successfully processed {result['student']}")
            print(f"Results saved in: {submissions_dir}/results.txt")
            print(f"Simulation outputs in: {submissions_dir}/sims.txt")
            if (submissions_dir / "logs.txt").exists():
                print(f"Errors logged in: {submissions_dir}/logs.txt")
    else:
        # Process multiple students
        student_dirs = [d for d in submissions_dir.iterdir() if d.is_dir()]
        
        if not student_dirs:
            print(f"No student directories found in {submissions_dir}")
            sys.exit(1)
        
        print(f"Found {len(student_dirs)} student submissions")
        
        # Process each submission (optionally in parallel)
        all_results = []
        student_dirs = sorted(student_dirs)
        num_workers = max(1, min(args.cores if args.cores else 1, len(student_dirs)))
        
        if num_workers > 1:
            print(f"Running with {num_workers} parallel processes")
            with mp.Pool(processes=num_workers) as pool:
                for result in pool.imap_unordered(process_student_submission, student_dirs):
                    all_results.append(result)
        else:
            for student_dir in student_dirs:
                result = process_student_submission(student_dir)
                all_results.append(result)
        
        # Generate summary report
        generate_summary_report(all_results, args.summary)
        print(f"Summary report written to {args.summary}")
        
        # Print quick summary
        successful = len([r for r in all_results if "error" not in r])
        failed = len(all_results) - successful
        print(f"Processing complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()
