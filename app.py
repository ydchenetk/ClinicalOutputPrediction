"""
Web UI for the Data Pipeline: Clean → Binary Model → Time-to-Response
"""
import os
import io
import json
import base64
import traceback
import threading
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect

# ── Pipeline imports ──────────────────────────────────────────────────
import importlib, sys

# Import the three pipeline modules
# The cleaning script has top-level pd.read_csv calls that may fail if data/ doesn't exist.
# We suppress those errors by pre-creating dummy dataframes, then import.
import types

def _safe_import(module_name):
    """Import a module by name, tolerating top-level read_csv failures."""
    try:
        return importlib.import_module(module_name)
    except (FileNotFoundError, Exception):
        # If top-level code fails (e.g. missing CSVs), load only the functions
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            # Try as a file path
            import pathlib
            path = pathlib.Path(module_name.replace("-", "-") + ".py")
            spec = importlib.util.spec_from_file_location(module_name, path)
        mod = types.ModuleType(module_name)
        mod.__file__ = str(spec.origin)
        # Read source, strip top-level statements that aren't defs/imports/classes
        with open(spec.origin, "r", encoding="utf-8") as f:
            source = f.read()
        # Execute with dummy dataframes to absorb read_csv errors
        dummy_ns = {
            "__name__": module_name,
            "__file__": str(spec.origin),
            "__builtins__": __builtins__,
        }
        try:
            exec(compile(source, spec.origin, "exec"), dummy_ns)
        except Exception:
            # Last resort: extract only function/class defs
            import ast, textwrap
            tree = ast.parse(source)
            safe_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
            safe_tree = ast.Module(body=safe_nodes, type_ignores=[])
            ast.fix_missing_locations(safe_tree)
            exec(compile(safe_tree, spec.origin, "exec"), dummy_ns)
        for k, v in dummy_ns.items():
            if callable(v) or isinstance(v, type):
                setattr(mod, k, v)
        sys.modules[module_name] = mod
        return mod

cleaning_mod = _safe_import("data-cleaning_analytical-dataset-pipeline")
binary_mod   = _safe_import("data-modeling_binary-prediction-pipeline")
survival_mod = _safe_import("data-modeling_time-to-response-pipeline")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────
jobs = {}  # job_id -> {status, progress, logs, results, error}


def _log(job_id, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append(f"[{ts}] {msg}")


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return encoded


def _run_pipeline(job_id, file_paths, steps, cleaning_cfg, binary_cfg, survival_cfg):
    """Background worker that runs the selected pipeline steps.
    file_paths: dict mapping domain/key -> absolute CSV path on disk.
    """
    try:
        jobs[job_id]["status"] = "running"
        results = {}

        # ── STEP 1: Data Cleaning ─────────────────────────────────────
        if "clean" in steps:
            _log(job_id, "Starting data cleaning …")
            jobs[job_id]["progress"] = 10

            config = {}
            domain_params = cleaning_cfg.get("domains", {})
            # File paths can come from top-level file_paths or from cleaning_cfg.filePaths
            clean_file_paths = cleaning_cfg.get("filePaths", {})
            all_paths = {**file_paths, **clean_file_paths}

            for domain, params in domain_params.items():
                fpath = all_paths.get(domain)
                if not fpath:
                    _log(job_id, f"Warning: no file path for domain '{domain}', skipping")
                    continue
                if not os.path.isfile(fpath):
                    _log(job_id, f"Warning: file not found for {domain}: {fpath}")
                    continue
                df = pd.read_csv(fpath)
                config[domain] = {"df": df, "params": params}

            first = cleaning_cfg.get("first", None)
            analytical_df = cleaning_mod.create_analytical_dataframe(config, first=first)

            out_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_analytical.csv")
            analytical_df.to_csv(out_path, index=False)

            results["cleaning"] = {
                "rows": len(analytical_df),
                "cols": len(analytical_df.columns),
                "columns": analytical_df.columns.tolist(),
                "csv_path": out_path,
                "head": analytical_df.head(10).to_html(classes="table table-sm table-striped", index=False),
            }
            _log(job_id, f"Cleaning done – {len(analytical_df)} rows × {len(analytical_df.columns)} cols")
            jobs[job_id]["progress"] = 33

        # ── STEP 2: Binary Prediction Model ───────────────────────────
        if "binary" in steps:
            _log(job_id, "Starting binary prediction modeling …")

            if "cleaning" in results:
                data = pd.read_csv(results["cleaning"]["csv_path"])
            elif "analytical_csv" in file_paths and os.path.isfile(file_paths["analytical_csv"]):
                data = pd.read_csv(file_paths["analytical_csv"])
            else:
                raise ValueError("Binary model requires cleaned data or an analytical CSV path.")

            (X_train, X_test, model_comp, fig_imp, df_imp,
             met_train, prob_train, fig_cab_train, fig_cm_train,
             met_test, prob_test, fig_cab_test, fig_cm_test) = binary_mod.prediction_process(data, binary_cfg)

            figures = {}
            for name, fig in [("importance", fig_imp),
                              ("calibration_train", fig_cab_train),
                              ("confusion_train", fig_cm_train),
                              ("calibration_test", fig_cab_test),
                              ("confusion_test", fig_cm_test)]:
                if fig is not None:
                    figures[name] = _fig_to_base64(fig)

            comp_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_model_comparison.csv")
            model_comp.to_csv(comp_path, index=False)

            results["binary"] = {
                "model_comparison": model_comp.to_html(classes="table table-sm table-striped", index=False),
                "feature_importance": df_imp.head(20).to_html(classes="table table-sm table-striped", index=False),
                "metrics_train": met_train if isinstance(met_train, dict) else str(met_train),
                "metrics_test": met_test if isinstance(met_test, dict) else str(met_test),
                "figures": figures,
            }
            _log(job_id, "Binary prediction modeling done.")
            jobs[job_id]["progress"] = 66

        # ── STEP 3: Time-to-Response Model ────────────────────────────
        if "survival" in steps:
            _log(job_id, "Starting time-to-response modeling …")

            # Analytical data is optional — user can choose to exclude it
            include_analytical = survival_cfg.pop("include_analytical", True)
            analytical = None
            if include_analytical:
                if "analytical_csv" in file_paths and os.path.isfile(file_paths["analytical_csv"]):
                    analytical = pd.read_csv(file_paths["analytical_csv"])
                elif "surv_analytical_csv" in file_paths and os.path.isfile(file_paths["surv_analytical_csv"]):
                    analytical = pd.read_csv(file_paths["surv_analytical_csv"])
                elif "cleaning" in results:
                    analytical = pd.read_csv(results["cleaning"]["csv_path"])
                else:
                    _log(job_id, "No analytical dataset found — proceeding without it.")

            if "pasi_csv" in file_paths and os.path.isfile(file_paths["pasi_csv"]):
                pasi = pd.read_csv(file_paths["pasi_csv"])
            elif "surv_pasi_csv" in file_paths and os.path.isfile(file_paths["surv_pasi_csv"]):
                pasi = pd.read_csv(file_paths["surv_pasi_csv"])
            else:
                raise ValueError("Survival model requires the raw PASI dataset.")

            train_res, train_ci, test_res, test_ci = survival_mod.survival_pipeline(pasi, analytical, survival_cfg)

            results["survival"] = {
                "train_c_index": float(train_ci),
                "test_c_index": float(test_ci),
                "train_head": train_res.head(10).to_html(classes="table table-sm table-striped", index=False),
                "test_head": test_res.head(10).to_html(classes="table table-sm table-striped", index=False),
            }
            _log(job_id, f"Survival modeling done – Train C-index: {train_ci:.4f}, Test C-index: {test_ci:.4f}")
            jobs[job_id]["progress"] = 100

        jobs[job_id]["status"] = "done"
        jobs[job_id]["results"] = results
        _log(job_id, "Pipeline finished successfully.")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = traceback.format_exc()
        _log(job_id, f"ERROR: {e}")


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train_redirect():
    return redirect("/train/cleaning")


@app.route("/train/cleaning")
def train_cleaning():
    return render_template("train_cleaning.html")


@app.route("/train/binary")
def train_binary():
    return render_template("train_binary.html")


@app.route("/train/survival")
def train_survival():
    return render_template("train_survival.html")


@app.route("/test")
def test_page():
    return render_template("test.html")


@app.route("/list-files", methods=["POST"])
def list_files():
    """List CSV files in a given directory path."""
    data = request.get_json()
    dir_path = data.get("path", "")
    if not dir_path or not os.path.isdir(dir_path):
        return jsonify({"error": f"Directory not found: {dir_path}"}), 400
    try:
        files = sorted([
            f for f in os.listdir(dir_path)
            if f.lower().endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))
        ])
        return jsonify({"files": files, "path": os.path.abspath(dir_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/columns", methods=["POST"])
def get_columns():
    """Return column names and a 5-row preview from a CSV file path."""
    data = request.get_json()
    file_path = data.get("path", "")
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 400
    try:
        df_preview = pd.read_csv(file_path, nrows=5)
        # Count rows efficiently without loading entire file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        return jsonify({
            "columns": df_preview.columns.tolist(),
            "preview": df_preview.to_html(classes="table table-sm table-striped table-bordered", index=False),
            "rows": max(row_count, 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/unique-values", methods=["POST"])
def get_unique_values():
    """Return unique values for a specific column in a CSV file."""
    data = request.get_json()
    file_path = data.get("path", "")
    column = data.get("column", "")
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 400
    if not column:
        return jsonify({"error": "No column specified"}), 400
    try:
        df = pd.read_csv(file_path, usecols=[column])
        values = df[column].dropna().unique().tolist()
        # Convert numpy types to native Python for JSON serialization
        values = [v.item() if hasattr(v, 'item') else v for v in values]
        return jsonify({"values": sorted(values, key=str)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/run", methods=["POST"])
def run_pipeline():
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0, "logs": [], "results": {}, "error": None}

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    steps = data.get("steps", [])
    if not steps:
        return jsonify({"error": "Select at least one pipeline step."}), 400

    # file_paths: domain -> absolute file path on disk
    file_paths = data.get("file_paths", {})
    cleaning_cfg = data.get("cleaning_cfg", {})
    binary_cfg = data.get("binary_cfg", {})
    survival_cfg = data.get("survival_cfg", {})

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, file_paths, steps, cleaning_cfg, binary_cfg, survival_cfg),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "logs": job["logs"],
        "error": job["error"],
        "results": job["results"],
    })


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_{filename}")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
