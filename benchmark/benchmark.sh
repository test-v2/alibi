#!/bin/bash -e

BRANCH_1=$1
EXPERIMENT_1=$2
BRANCH_2=$3
EXPERIMENT_2=$4

checkout_error()
{
  BRANCH=$1
  echo "Could not checkout $BRANCH"
  echo "Ensure the branch or commit hash exist and all changes on current branch are committed!!!"
  echo "Exiting"
} >&2

experiment_error()
{
  BRANCH=$1
  YAML=$2
  echo "Could not run expermient on $BRANCH"
  echo "Ensure the configuration $YAML exists in benchmark/configs/ or check your expermient code!"
  echo "Exiting ..."
} >&2

echo "Checking out ${BRANCH_1}..."
git checkout "$BRANCH_1" || { checkout_error "$BRANCH_1"; exit 1;}

echo "Runnig expermient with configuration ${EXPERIMENT_1} on this branch ..."
python benchmark/experiment.py --config "benchmark/configs/$EXPERIMENT_1" ||
 { experiment_error "$BRANCH_1" "$EXPERIMENT_1"; exit 1;}

echo "Checking out ${BRANCH_2}..."
git checkout "$BRANCH_2" || { checkout_error "$BRANCH_2"; exit 1;}

python benchmark/experiment.py --config "benchmark/configs/$EXPERIMENT_2" ||
 { experiment_error "$BRANCH_2" "$EXPERIMENT_2"; exit 1;}
echo "Benchmarking complete!"