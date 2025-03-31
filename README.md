# OffSIDES

The Off-label Side Effect Resource (OffSIDES) is a collection of statistically associated adverse drug reactions derived from the analysis of adverse drug event reports submitted to the FDA's Adverse Event Reporting System (FAERS). OffSIDES uses a data-driven method to correct for common biases and sources of noise that limit traditional analyses of these data. The method used is called Statistical CorRection of Uncharacterized Bias (SCRUB) and uses high-dimensional propensity score matching to mitigate confounding biases. The companion method, Latent Signal Detection (LSD), can infer the presence of adverse reactions when direct evidence or reporting is  unavailable. These methods are combined together to produce a set of hypothesees adverse drug reactions. Known side effects  of drugs that are already reported on the structured product label, for example, are filtered out so that only those that are "off-label" are included in this resource. For known drug side effects, see the [OnSIDES resource](http://github.com/tatonetti-lab/onsides). 

The process of creating the OffSIDES resource is broken down into the following steps:

### Building the database
1. Download and process the latest FAERS data (e.g. from opendata.fda.gov)
 - these scripts assume that the data are in a SQL database
 - also some derivative (non-FDA) tables are necessary, see FAERSDB.md for details
2. Choose a date range and strata to run the anlaysis on.
 - each script takes as input --start_year and --end_year parameters
3. Build confounding matrices
 - run `python3 src/build_confounding_matrices.py --start_year 2004 --end_year 2004`
3. Run high-dimensional propensity score matching on the drug report.
 - run `python3 src/hdpsm.py --start_year 2004 --end_year 2004`
4. Generate disproportionality statistics
 - run `python3 src/est_assoc_stats.py --start_year 2004 --end_year 2004`

### Evaluation
5. Identify and build confounded reference set
 - run notebook named `biased_by_confounders.ipynb`
6. Evaluate on reference sets:
 - run notebook named `evaluate_performance_confounding.ipynb`
 - run notebook named `evaluate_performance_onsides.ipynb`

## Required Files
- OnSIDES Warnings and Precautions table

## Optional Files
- Map from drug indication to snomed




