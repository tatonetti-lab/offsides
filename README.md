# OffSIDES

The Off-label Side Effect Resource (OffSIDES) is a collection of statistically associated adverse drug reactions
derived from the analysis of adverse drug event reports submitted to the FDA's Adverse Event Reporting
System (FAERS). OffSIDES uses a data-driven method to correct for common biases and sources of noise that
limit traditional analyses of these data. The method used is called Statistical CorRection of Uncharacterized Bias
(SCRUB) and uses high-dimensional propensity score matching to mitigate confounding biases. The companion method,
Latent Signal Detection (LSD), can infer the presence of adverse reactions when direct evidence or reporting is 
unavailable. These methods are combined together to produce a set of hypothesees adverse drug reactions. Known side effects 
of drugs that are already reported on the structured product label, for example, are filtered out so that only
those that are "off-label" are included in this resource. For known drug side effects, see the 
[OnSIDES resource](http://github.com/tatonetti-lab/onsides). 

The process of creating the OffSIDES resource is broken down into the following steps:

1. Download and process the latest FAERS data from opendata.fda.gov (see faers_processor.py). 
2. Compile a "dataset" from the processed data (see faers_compile_dataset.py)
3. Run high-dimensional propensity score matching on the drug reports (see propensity_score_match.py)
4. Generate disproportionality statistics (TBW)


