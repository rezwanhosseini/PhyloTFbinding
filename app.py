from flask import Flask, render_template, request
from pathlib import Path
import subprocess
import re
from config import SCRIPT1_PATH, PFM_PATH, FASTA_DIR, OUTPUT_BASE_DIR, SCRIPT2_PATH

from datetime import datetime

app = Flask(__name__)

def is_valid_gene_name(gene: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", gene))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
#def run():
#    gene = request.form.get("gene", "").strip()
#    method = request.form.get("method", "").strip()
#    phenotype = request.form.get("phenotype", "").strip()

#    fasta_file = FASTA_DIR / f"{gene}_combined_names.fa"

#    if not fasta_file.exists():
#       return render_template(
#           "result.html",
#           error=f"{gene} not exist in out protein-coding list.",
#           gene=gene,
#           method=method,
#           phenotype=phenotype
#       )


def run_analysis():
    gene = request.form.get("gene", "").strip()
    motif = request.form.get("motif", "").strip()

    if not gene:
        return render_template("result.html", error="Gene name is required.")

    if not is_valid_gene_name(gene):
        return render_template("result.html", error="Invalid gene name.")

    fasta_path = FASTA_DIR / f"{gene}_combined_names.fa"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT_BASE_DIR / gene / motif / run_id

    if not fasta_path.exists():
        return render_template(
            "result.html",
            error=f"FASTA file not found: {fasta_path}"
        )

    if not motif:
        return render_template(
            "result.html",
            error=f"motif is required."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    cmd1 = [
        "python3",
        str(SCRIPT1_PATH),
        "--pfm", str(PFM_PATH),
        "--fasta", str(fasta_path),
        "--motif", str(motif),
        "--outdir", str(outdir)
    ]

    result1 = subprocess.run(
        cmd1,
        capture_output=True,
        text=True
    )

#    return render_template(
#        "result.html",
#        gene=gene,
#        motif=motif,
#        command=" ".join(cmd1),
#        returncode=result.returncode,
#        stdout=result.stdout,
#        stderr=result.stderr,
#        outdir=str(outdir)
#    )

    if result1.returncode != 0:
        return render_template(
            "result.html",
            error="Phase 1 failed",
            stdout=result1.stdout,
            stderr=result1.stderr
        )

    print("Phase 1 done!")


# add the code to run phylo_heatmap_visionloss.R with the output 
    motifhit_summary = outdir / "pwm_summary.csv"
    motifhit_scores = outdir / "pwm_results.h5"
    orderby = request.form.get("orderby", "").strip()

    print("Checking files exist:")
    print(motifhit_summary.exists(), motifhit_summary)
    print(motifhit_scores.exists(), motifhit_scores)

    cmd2 = [
        "Rscript",
        str(SCRIPT2_PATH),
        "--gene", str(gene),
        "--motif", str(motif),
        "--runid", str(run_id),
        "--summary", str(motifhit_summary),
        "--h5", str(motifhit_scores),
        "--orderby", str(orderby)
    ]
    result2 = subprocess.run(
        cmd2,
        capture_output=True,
        text=True
    )
    print("Phase 2 return code:", result2.returncode)
    print("Phase 2 STDOUT:")
    print(result2.stdout)
    print("Phase 2 STDERR:")
    print(result2.stderr)

    if result2.returncode != 0:
        return render_template(
            "result.html",
            error="Phase 2 (plotting) failed.",
            stdout2=result2.stdout,
            stderr2=result2.stderr,
            gene=gene,
            motif=motif
        )

    return render_template(
        "result.html",
        gene=gene,
        motif=motif,
#        command=" ".join(cmd2),
#        returncode=result1.returncode,
        stdout1=result1.stdout,
        stdout2=result2.stdout,
        stderr1=result1.stderr,
        stderr2=result2.stderr,
        returncode=result2.returncode,
        outdir=str(outdir)
    )


    print("Phase 2 done!")


if __name__ == "__main__":
    app.run(debug=True)

