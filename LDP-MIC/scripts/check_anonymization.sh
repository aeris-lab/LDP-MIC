#!/bin/bash
# check_anonymization.sh
# Run this script before uploading to check for identifying information

set -e

echo "=========================================="
echo "  LDP-MIC Anonymization Checker"
echo "=========================================="
echo ""

FOUND_ISSUES=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pattern() {
    local pattern="$1"
    local description="$2"
    local result=$(grep -r -i "$pattern" --include="*.py" --include="*.ipynb" --include="*.md" --include="*.sh" --include="*.yaml" --include="*.yml" --include="*.slurm" . 2>/dev/null | grep -v "check_anonymization.sh" | grep -v ".git/")
    
    if [ -n "$result" ]; then
        echo -e "${RED}[FOUND]${NC} $description"
        echo "$result" | head -10
        if [ $(echo "$result" | wc -l) -gt 10 ]; then
            echo "  ... and more matches"
        fi
        echo ""
        FOUND_ISSUES=1
    else
        echo -e "${GREEN}[OK]${NC} $description"
    fi
}

echo "--- Checking for HPC System Names ---"
check_pattern "frontier" "Frontier supercomputer"
check_pattern "summit" "Summit supercomputer"
check_pattern "perlmutter" "Perlmutter supercomputer"
check_pattern "stampede" "Stampede supercomputer"

echo ""
echo "--- Checking for Institution Names ---"
check_pattern "oak ridge" "Oak Ridge National Laboratory"
check_pattern "ornl" "ORNL abbreviation"
check_pattern "olcf" "OLCF"
check_pattern "argonne" "Argonne National Laboratory"
check_pattern "berkeley" "Berkeley Lab"
check_pattern "nersc" "NERSC"

echo ""
echo "--- Checking for System Paths ---"
check_pattern "/lustre/" "Lustre filesystem paths"
check_pattern "/gpfs/" "GPFS filesystem paths"
check_pattern "/ccs/" "CCS paths"
check_pattern "/orion/" "Orion paths"
check_pattern "csc569" "Project codes"

echo ""
echo "--- Checking for User Info ---"
check_pattern "ywj17593" "Username"
check_pattern "@ornl" "ORNL email domain"
check_pattern "@gov" "Government email domain"

echo ""
echo "--- Checking for Email Patterns ---"
result=$(grep -r -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(edu|gov|org|com)" --include="*.py" --include="*.ipynb" --include="*.md" . 2>/dev/null | grep -v ".git/" | grep -v "example@" | grep -v "user@")
if [ -n "$result" ]; then
    echo -e "${RED}[FOUND]${NC} Email addresses"
    echo "$result"
    echo ""
    FOUND_ISSUES=1
else
    echo -e "${GREEN}[OK]${NC} Email addresses"
fi

echo ""
echo "--- Checking for GPU Hardware Specifics ---"
check_pattern "mi250" "AMD MI250X GPU"
check_pattern "mi300" "AMD MI300 GPU"

echo ""
echo "--- Checking Jupyter Notebooks ---"
if [ -d "notebooks" ]; then
    for notebook in $(find notebooks -name "*.ipynb" 2>/dev/null); do
        if grep -q '"username"' "$notebook" 2>/dev/null; then
            echo -e "${RED}[FOUND]${NC} Username in notebook: $notebook"
            FOUND_ISSUES=1
        fi
        if grep -q '"hostname"' "$notebook" 2>/dev/null; then
            echo -e "${RED}[FOUND]${NC} Hostname in notebook: $notebook"
            FOUND_ISSUES=1
        fi
    done
    echo -e "${GREEN}[DONE]${NC} Notebook metadata check"
else
    echo -e "${YELLOW}[SKIP]${NC} No notebooks directory"
fi

echo ""
echo "=========================================="
if [ $FOUND_ISSUES -eq 1 ]; then
    echo -e "${RED}Issues found! Review and fix before uploading.${NC}"
    exit 1
else
    echo -e "${GREEN}No identifying information found!${NC}"
    echo ""
    echo "Recommended additional manual checks:"
    echo "  - Review any TODO/FIXME comments"
    echo "  - Check figure metadata in saved images"
    echo "  - Verify git history is clean"
    exit 0
fi
