#!/bin/bash
# Local Evaluator for Entrega 2
# Usage: ./local_evaluator.sh rf004

RF=${1:-all}

# 1. Create evaluator folder
mkdir -p .evaluator

# 2. Define resources
BASE_URL="https://raw.githubusercontent.com/MISW-4301-Desarrollo-Apps-en-la-Nube/recursos-evaluador/main/entrega2"
FILES=(
    "verify_old_endpoints.json"
    "evaluate_rf003.json"
    "evaluate_rf003_consistency.json"
    "evaluate_rf004.json"
    "evaluate_rf004_consistency.json"
    "evaluate_rf005.json"
    "evaluate_rf005_consistency.json"
)

# 3. Download resources
echo "Downloading test collections..."
for FILE in "${FILES[@]}"; do
    if [ ! -f ".evaluator/$FILE" ]; then
        curl -sSL -o ".evaluator/$FILE" "$BASE_URL/$FILE"
    fi
done

# 4. Get BASE_PATH from config.yaml
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found!"
    exit 1
fi

# Try to use yq if available, otherwise use grep/sed
if command -v yq &> /dev/null; then
    BASE_PATH=$(yq '.url' config.yaml)
else
    BASE_PATH=$(grep 'url:' config.yaml | sed 's/url: //;s/"//g;s/ //g')
fi

if [ -z "$BASE_PATH" ] || [ "$BASE_PATH" == "null" ]; then
    echo "❌ Could not find 'url' in config.yaml"
    exit 1
fi

echo "✅ Target URL: $BASE_PATH"

# 5. Run tests
run_newman() {
    echo "🚀 Running tests: $1"
    npx newman run ".evaluator/$1" --env-var "BASE_PATH=$BASE_PATH" --verbose --insecure
}

if [[ "$RF" == "all" || "$RF" == "rf003" ]]; then
    run_newman "evaluate_rf003.json"
fi

if [[ "$RF" == "all" || "$RF" == "rf004" ]]; then
    run_newman "evaluate_rf004.json"
fi

if [[ "$RF" == "all" || "$RF" == "rf005" ]]; then
    run_newman "evaluate_rf005.json"
fi

echo "✅ Evaluation finished."
