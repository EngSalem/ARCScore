## This script runs the ARCScore generation process from the command line.
set -e 




## ARCSCORE engine attributes
BASE_MODEL_URL="https://api.openai.com/v1" # Adjust if using a local model
API_KEY="API_KEY_HERE" # Replace with your actual API key
EVAL_MODEL="gpt-4o-mini"





echo "=============================="
echo " Generating ARCScore Using ${EVAL_MODEL}"
echo "=============================="

SUMMARIES_DIR="/path/to/summaries" # Directory containing generated summaries # assumed in pkl format
SUMM_MODEL="llama3.1-8B-Instruct" # Model used to generate summaries
echo "Summaries directory: $SUMMARIES_DIR"


if [ ! -d "$SUMMARIES_DIR" ]; then
    echo "ERROR: Summaries directory not found: $SUMMARIES_DIR"
    exit 1
fi

OUTPUT_DIR="" # output directory for ARCScore results
mkdir -p "$(dirname "$OUTPUT_DIR")"


python generate_arcscores_from_pickle.py \
               --pkl "$SUMMARIES_DIR/generated_summaries.pkl" \
               --model-name "$SUMM_MODEL" \
               --out-dir "$OUTPUT_DIR" \
               --arcscore-api-key "$API_KEY" \
               --arcscore-base-url "$BASE_MODEL_URL" \
               --arcscore-model "$EVAL_MODEL" \
               --batch-size 50 \ # number of concurrent facts to score
               --num-articles 100 ## full CANLII100

echo "ALL ARCScore GENERATION RUNS COMPLETE."






