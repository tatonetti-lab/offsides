year=$1
echo "time ("
echo python3 src/build_confounding_matrices.py --start_year $year --end_year $year
echo python3 src/hdpsm.py --start_year $year --end_year $year
echo python3 src/est_assoc_stats.py --start_year $year --end_year $year
echo python3 src/build_confounded_datasets.py --start_year $year --end_year $year
echo ")"