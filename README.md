# Career Progression and Promotion Gap Analysis for Retention Optimization
### A Career-Trajectory Intelligence Project for Palo Alto Networks | Unified Mentor Internship

This repository contains an end-to-end HR analytics project that looks at employee
attrition from a different angle than most standard approaches. Instead of asking
"who is likely to leave," it asks "what structural career problems are pushing
someone toward leaving, even before they show any obvious signs of disengagement."
The project takes a raw HR dataset, engineers career-progression features, groups
employees into career-path segments using two independent clustering methods,
scores each employee's promotion-gap risk, and surfaces a specific, prioritized
list of employees who would benefit from an HR intervention right now, along with
a concrete suggested action for each one.

Before you read further: this README is written as a walkthrough, not a
reference manual. Each step below explains what was done and why, shows the
actual output or chart produced by that step, and then explains what that
output means. If you want to see the raw code instead of the narrative, the
three Jupyter notebooks in this repository contain the full, runnable version
of every step described here.
---

## Table of Contents

- [Situation](#situation)
- [Task](#task)
- [Action: Step-by-Step Walkthrough](#action-step-by-step-walkthrough)
  - [Step 1: Exploratory Data Analysis](#step-1-exploratory-data-analysis)
  - [Step 2: Feature Engineering and Preprocessing](#step-2-feature-engineering-and-preprocessing)
  - [Step 3: Career Path Clustering and Risk Scoring](#step-3-career-path-clustering-and-risk-scoring)
  - [Step 4: The Interactive Streamlit Dashboard](#step-4-the-interactive-streamlit-dashboard)
- [Result](#result)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How to Run This Project](#how-to-run-this-project)
- [Key Findings](#key-findings)
- [Limitations and Honest Caveats](#limitations-and-honest-caveats)
- [Possible Future Improvements](#possible-future-improvements)
- [Dataset Source](#dataset-source)
- [Acknowledgments](#acknowledgments)

---

## Situation

Most attrition-prediction work in HR analytics answers a narrow question: given
an employee's data, what is the probability they leave the company. That is a
useful number, but it is also an incomplete one. It tells an HR team who might
walk out the door without explaining why they were on their way out in the
first place. In practice, this leaves HR teams reacting to attrition with
generic retention offers (a raise here, a vague promise of career growth there)
that are not actually targeted at the real underlying problem for that specific
person.

Palo Alto Networks, like most large organizations, faces this exact gap.
Employees rarely leave because of one dramatic event. More often, they leave
after a long, quiet accumulation of unmet expectations: years without a
promotion, too long in the same role without meaningful change, minimal
investment in training, or a string of different managers with no real
continuity. None of these things show up as a single alarming metric. They
build up slowly, and by the time attrition data reflects them, the employee has
usually already left.

This project was built to close that specific gap: to move from reactive
attrition prediction toward proactive career-trajectory intelligence.

## Task

The objective was to design and build a complete analytics pipeline and
dashboard that could:

1. Quantify each employee's career health using measurable, tenure-normalized
   indicators rather than raw year-counts that are not comparable across people
   with very different lengths of service.
2. Group employees into meaningful career-trajectory segments using
   unsupervised machine learning, and validate that segmentation with a second,
   independent clustering method rather than trusting a single algorithm's
   output blindly.
3. Score each employee's promotion-gap risk on a simple Low, Medium, High scale
   that a non-technical HR stakeholder could immediately understand and act on.
4. Identify a specific, prioritized list of Retention Opportunity employees:
   people who have not yet left and are not already flagged as high attrition
   risks in a conventional sense, but who show early structural warning signs
   of disengagement.
5. Turn all of this into an interactive tool that department heads and HR
   business partners could actually use themselves, with filters, drill-downs,
   and a simulator, rather than a static one-time report that goes stale the
   moment it is delivered.

With the goal defined, the next section walks through exactly how each part
was built, in the order it was actually done.

---

## Action: Step-by-Step Walkthrough

### Step 1: Exploratory Data Analysis

**What was done and why.** Before any modeling, the raw dataset needed to be
checked rather than trusted. This step loads the raw HR CSV (1,470 employee
records, 31 original fields) and immediately checks its shape, its missing
values, and its duplicate rows. Skipping this is the most common reason a
downstream model quietly breaks: if there were missing values or duplicate
records, every later step would inherit that problem without any obvious
warning sign.

![image alt](https://github.com/user-attachments/assets/13a9ebe3-124c-4a00-9508-2917e8c4dcd8)

The output above confirms a clean starting point: 1,470 rows, 31 columns, zero
missing values, and zero duplicate rows. This is what let the rest of the
project proceed without needing an extra data-cleaning stage for basic
integrity issues.

**Attrition distribution.** With the data confirmed clean, the next question
was the most fundamental one: how many employees actually leave. This was
plotted two ways, a bar chart for exact counts and a donut chart for
proportion, specifically because a raw count and a percentage answer slightly
different intuitions for a reader.

![image alt](https://github.com/user-attachments/assets/8458df9d-2f00-481e-8b77-0e0083d150c6)

The result shows roughly 84 percent of employees stayed and 16 percent left.
This number matters beyond curiosity: it reveals that the dataset is
imbalanced, meaning "left" is a comparatively rare event. That fact directly
shaped a later design decision, covered in Step 3, to base the Retention
Opportunity logic on promotion-gap risk rather than on raw attrition alone,
since raw attrition alone is too rare an event to build a useful prioritization
system on.

**Correlation heatmap.** Next, every numeric field in the dataset was checked
against every other numeric field, and against Attrition specifically, using a
correlation heatmap.

![image alt](https://github.com/user-attachments/assets/89506c62-fec1-48a6-b5a2-410ee6d2a8cf)

The reasoning here was twofold. First, this shows which raw numeric fields
relate most to attrition before any new features are engineered, giving a
baseline to compare against later. Second, it flags multicollinearity, meaning
two features that essentially measure the same thing, which would be wasteful
or misleading to feed twice into the same clustering model. The actual result
was that no single numeric feature correlated strongly with attrition; the
strongest correlations sat below 0.2. That outcome is not a failure of the
analysis, it is the core justification for the entire project: if attrition
were explainable by one variable, this deeper structural analysis would not be
necessary.

**Outlier screening on tenure variables.** The five tenure-related fields that
the rest of this project depends on (YearsAtCompany, YearsInCurrentRole,
YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears) were plotted
as boxplots to visually check for outliers.

![image alt](https://github.com/user-attachments/assets/84efd53f-d79d-47b1-8ab0-c09583f95d3d)

This step exists because the clustering algorithm used in Step 3, K-Means,
measures straight-line distance between employees. A handful of employees with
extremely long tenure could sit far away from everyone else in that distance
space and quietly pull the clustering result toward themselves, distorting the
grouping for the other 98 percent of the workforce. Seeing the outliers here,
visible as dots above the upper whisker in the boxplots, is what justified
actually removing a conservative set of them in Step 2.

**Attrition by department and job role.** Correlation only works on numeric
fields, so a separate view was needed for the categorical fields Department and
JobRole: group by each one, and calculate the percentage who left within each
group.

![image alt](https://github.com/user-attachments/assets/02b5164c-8fa6-421c-9f63-a37739b02870)

This showed that attrition is not evenly spread across the organization.
Certain individual-contributor roles and the Sales department showed
noticeably higher attrition than senior or managerial roles, which is a
pattern that gets revisited later through the lens of promotion-gap risk
rather than raw attrition.

**Feature engineering: the four career-progression ratios.** This is the
single most important step in the entire notebook. Four new columns were
created, each expressing a career signal as a ratio rather than a raw count:

- Promotion Gap Ratio, calculated as YearsSinceLastPromotion divided by
  YearsAtCompany
- Role Stagnation Index, calculated as YearsInCurrentRole divided by
  YearsAtCompany
- Training Intensity Score, calculated as TrainingTimesLastYear divided by
  YearsAtCompany
- Manager Stability Indicator, calculated as YearsWithCurrManager divided by
  YearsAtCompany

The reasoning is that a raw number like "three years since the last
promotion" means something completely different for a four-year employee (75
percent of their entire time at the company without a promotion, a real
warning sign) than for a twenty-year employee (15 percent of their tenure,
entirely normal). Dividing by tenure puts every employee on the same
comparable scale, roughly zero to one, regardless of how long they have been
at the company.

![image alt](https://github.com/user-attachments/assets/228e44f9-4b7d-4b3f-910c-915e6a96c937)

The histograms above confirm the ratios behave sensibly: mostly clustered in
the zero-to-one range, right-skewed, meaning most employees look healthy on
these measures while a smaller group shows meaningfully higher stagnation.
That is exactly the shape needed before trusting these features enough to feed
them into a clustering model.

---

### Step 2: Feature Engineering and Preprocessing

**Encoding categorical fields.** Clustering algorithms and most statistical
models cannot read text directly, so text fields like Department, JobRole,
Gender, MaritalStatus, BusinessTravel, EducationField, and OverTime were
converted into numeric code columns using label encoding. Crucially, the
original text columns were kept alongside the new numeric ones, so the
dashboard built later could still display a human-readable label like "Sales"
instead of a meaningless number.

![image alt](https://github.com/user-attachments/assets/48d0cbca-ce31-42bf-ba70-a263c13eb5a7)

This chart was generated as a visual reference at the moment of encoding,
showing both the raw counts and the proportional share of two of the more
business-relevant categories, Department and Marital Status. The purpose was
to have a clear record of what each numeric code actually represents before
that context gets lost deeper in the pipeline.

**Outlier removal.** Building on the outlier screening from Step 1, this stage
actually removed the extreme cases, using an IQR-based rule applied to
YearsAtCompany and TotalWorkingYears. A stricter-than-standard threshold was
used deliberately, so that only genuinely extreme edge cases were dropped
rather than ordinary senior employees who simply have long, legitimate tenure.

![image alt](https://github.com/user-attachments/assets/4c0b0419-903e-4276-9536-9b4652088e1c)

The result removed 19 records out of 1,470, roughly 1.3 percent of the
dataset. The chart above shows this two ways: a donut chart of records kept
versus removed, and a side-by-side boxplot of TotalWorkingYears before and
after removal, so the effect is visible rather than just stated as a number.
Removing data should always be something you can see and justify, not a silent
step buried in code.

---

### Step 3: Career Path Clustering and Risk Scoring

**Scaling the features.** Six features were selected for clustering:
Promotion Gap Ratio, Role Stagnation Index, Training Intensity Score, Manager
Stability Indicator, YearsAtCompany, and TotalWorkingYears. Before clustering,
these were standardized using StandardScaler, which rescales every column to
have a mean of zero and a standard deviation of one.

![image alt](https://github.com/user-attachments/assets/810b5f4d-dc33-4650-9919-6e0355c71306)

This step matters because, without it, YearsAtCompany, which ranges roughly
from zero to forty, would completely dominate the distance calculation over
Promotion Gap Ratio, which ranges from zero to one, simply because its raw
numbers are larger, not because it is actually more important.

**K-Means clustering.** Employees were grouped into four clusters using
K-Means, chosen to match the four career archetypes described in the original
project brief: fast-trackers, senior contributors, early-career explorers, and
promotion-stalled employees.

**K-Means silhouette output: 0.306**

The resulting silhouette score, a measure from negative one to one of how
distinct the clusters are from each other, came out to approximately 0.306,
which is a typical and reasonable result for real-world HR data.

**Validating with Hierarchical clustering.** K-Means starts from random
initial cluster centers, which means, in principle, its result could reflect a
quirk of where it started rather than a genuine pattern in the data. To check
this, an entirely different algorithm, Agglomerative Hierarchical clustering
with Ward linkage, was run independently on the same data, and its output was
compared to the K-Means result using the Adjusted Rand Index.

![image alt](https://github.com/user-attachments/assets/82df74d5-da1b-4cda-8da6-c212a789994a)

The dendrogram above visualizes how hierarchical clustering progressively
merged similar employees, on a random sample of 120 employees purely for
visual readability. The Adjusted Rand Index between the two algorithms came
out to 0.725, meaning strong agreement, since a score of 1.0 represents
perfect agreement and 0.0 represents no better than random chance. This is the
evidence that the four-segment structure found by K-Means is a real pattern in
the data, not an artifact of one algorithm's starting point.

**Naming the clusters dynamically.** K-Means only outputs cluster numbers,
zero through three; it has no concept of what those numbers mean in business
terms. Rather than hardcoding a fixed mapping from cluster number to name,
which would silently break if the underlying data changed, each cluster was
ranked by its actual average promotion-gap ratio and tenure, and named based
on that real behavior.

![image alt](https://github.com/user-attachments/assets/faa52cfe-6fc7-40f7-b2cc-3d85bb531cc3)

The four resulting segments were Steady Performers and Fast-Trackers (721
employees, the largest group), Promotion-Stalled and Stagnant Profiles (279
employees), Long-term Contributors and Senior Staff (240 employees), and
Early-Career Explorers and New Joiners (211 employees).

**Understanding what makes each cluster distinct.** The bar chart above shows
cluster size, but not cluster character. To see what actually distinguishes
each segment, a heatmap was built comparing every cluster's average value
across all six clustering features side by side.

![image alt](https://github.com/user-attachments/assets/3ce6bb83-6502-421c-9286-da8641c20bb7)

This is arguably the single most explanatory chart in the whole project. It
shows, at a glance, that the Promotion-Stalled segment runs hot (colored dark
red) specifically on Promotion Gap Ratio and Role Stagnation Index, while the
Early-Career Explorers segment runs cold (dark blue) on almost every tenure
related measure simply because they have not been at the company long enough
to accumulate those signals yet.

**Promotion Gap Risk scoring.** Each employee was assigned a Low, Medium, or
High risk tier. The base tier came from simple thresholds on Promotion Gap
Ratio, and then, as required by the original project brief, this was combined
with cluster membership: anyone sitting in the Promotion-Stalled cluster with
a borderline Medium ratio was escalated to High, because their cluster
membership reflects compounding stagnation signals that the ratio alone would
miss.

![Promotion gap risk distribution](screenshots/14_risk_distribution_output.png)

The result was 579 employees at Low risk, 427 at Medium, and 445 at High risk,
meaning just over 30 percent of the workforce falls into the highest risk
tier.

**Training Need Indicator and Manager Stability Impact.** Two further
indicators were computed. The Training Need Indicator flags employees whose
training participation sits in the bottom third of the company and who also
show meaningful role stagnation, targeting people who are both under-developed
and stuck, rather than simply under-trained. The Manager Stability Impact
compares attrition rates between employees with high versus low manager
continuity.

![image alt](https://github.com/user-attachments/assets/795c34b9-5571-4599-a96c-4e0a930466c7)
![image alt](https://github.com/user-attachments/assets/71f293fd-6c91-4540-b849-6d47cd76dd31)

The chart above shows a real, measurable difference: employees with low
manager continuity left at 17.7 percent, compared to 14.5 percent for
employees with high manager continuity. This is direct evidence that manager
churn is not a neutral factor, it correlates with people actually leaving.

**Retention Opportunity identification.** This is the headline output of the
entire project. Employees were flagged as High-Priority Retention
Opportunities if they had not already left the company and were scored at
High promotion-gap risk, meaning they are showing early structural warning
signs before any conventional attrition indicator would catch them. Each
flagged employee was then given a specific, rule-based suggested action,
drawn from whichever underlying signal was driving their risk: enroll in
training, consider a role rotation, schedule a promotion review, or review
manager continuity.

![image alt](https://github.com/user-attachments/assets/f7326bb4-731c-4139-b68e-c61f5d44e8a6)
![image alt](https://github.com/user-attachments/assets/44582844-4c0d-4069-8f82-3a95ecc75c7f)

The result identified 366 employees, roughly one in four of the workforce that
has not already left, as High-Priority Retention Opportunities. This is the
number that turns the whole analysis from an academic exercise into something
an HR team could act on directly, starting tomorrow.

---

### Step 4: The Interactive Streamlit Dashboard

All of the analysis above is reproducible in three notebooks, but a notebook
is not something a department head or HR business partner is going to open
and run themselves. The final step was to package everything into an
interactive Streamlit dashboard with five modules.

**Overview and Career Path Clustering.** The landing view shows top-level KPI
cards and an auto-generated set of plain-language insights, followed by a tab
where the four career segments can be explored interactively, including a 3D
scatter plot of stagnation versus experience.

![image alt](https://github.com/user-attachments/assets/18ccfc91-2298-4d5b-8d02-312368c4442b)

**Promotion Gap Monitor.** This tab lets a user drill from Department, into
Job Role, into Risk tier, using an interactive sunburst chart, to see exactly
where promotion stagnation concentrates in the organization.

![image alt](https://github.com/user-attachments/assets/066cd59e-17b2-4240-b898-4f86bd1e2bfa)

**Retention Opportunity Panel.** This tab surfaces the 366 high-priority
employees directly, along with a what-if promotion-intervention simulator that
lets a user drag a slider to see how promoting a given percentage of high-risk
employees would reduce overall risk exposure.

![image alt](https://github.com/user-attachments/assets/b1d1852b-28cb-411a-8d10-3631c8119fda)

**Managerial Insight Dashboard.** This tab isolates the manager-continuity
finding from Step 3 and extends it to a team level, showing which departments
have the highest concentration of low manager continuity alongside their
attrition rates.

![image alt](https://github.com/user-attachments/assets/2c6f12bc-5773-4378-8691-6441ae353c1b)

**Target Export.** The final tab presents the full, sortable roster of
high-priority employees, including each individual's specific suggested
action, with a one-click CSV export so an HR team can take the list directly
into their own workflow.

![image alt](https://github.com/user-attachments/assets/dffe16b7-fbf0-4a5d-b901-31d55cfd54f4)

---

## Result

The pipeline produced four validated career segments across 1,451 employees
after outlier removal. The K-Means and Hierarchical clustering results agreed
strongly, with an Adjusted Rand Index of 0.725, giving real confidence that
these segments reflect genuine structure in the data rather than a modeling
coincidence.

Of the full workforce, 30.7 percent were scored as High promotion-gap risk.
Within that group, 366 employees were identified as High-Priority Retention
Opportunities, each with a specific suggested action rather than a generic
flag. Separately, the analysis confirmed a real link between manager
continuity and attrition, with a 3.2 percentage point gap in attrition rate
between high and low manager-continuity employees.

The end deliverable is not just a set of findings but a reusable,
re-runnable system: a person could bring in next quarter's HR data, run the
same three notebooks, and get a freshly updated, freshly labeled segmentation
and priority list without touching any code.

---


## Tech Stack

- **Python** for the entire pipeline
- **pandas / numpy** for data manipulation and feature engineering
- **scikit-learn** for StandardScaler, KMeans, AgglomerativeClustering, and
  clustering evaluation metrics (silhouette score, Adjusted Rand Index)
- **scipy** for hierarchical clustering linkage and dendrogram construction
- **matplotlib / seaborn** for the static charts embedded in the notebooks
- **plotly** for the interactive charts inside the Streamlit dashboard
- **Streamlit** for the interactive dashboard application
- **Jupyter Notebook** for the documented, step-by-step analysis


## Key Findings

- The dataset shows a 16.1 percent overall attrition rate, a typical level of
  class imbalance for HR data, which shaped how the Retention Opportunity
  logic was designed, since raw attrition alone is too rare an event to be the
  only signal used for prioritization.
- No single numeric feature correlates strongly with attrition on its own,
  with the strongest individual correlations sitting below 0.2, supporting
  the project's central premise that attrition here is driven by a
  combination of structural career factors rather than any one variable
  acting in isolation.
- Four career segments emerged with meaningfully different profiles. The
  Promotion-Stalled segment, at 279 employees, shows an average promotion gap
  ratio of 0.85, compared to 0.12 for the largest segment, Steady Performers
  and Fast-Trackers, which contains 721 employees.
- Manager continuity has a measurable relationship with attrition. Employees
  with low manager continuity left at 17.7 percent, versus 14.5 percent for
  employees with high manager continuity.
- 366 employees, roughly one in four of the non-attrited workforce, were
  identified as High-Priority Retention Opportunities requiring some form of
  career intervention before they reach a point of active disengagement.

## Limitations and Honest Caveats

This project uses rule-based thresholds in several places rather than
statistically tuned or model-derived cutoffs. The Low, Medium, High
promotion-gap risk bands, and the threshold used for the Training Need
Indicator, were chosen as reasonable, explainable defaults rather than
optimized against any external validation criterion, because no ground truth
exists for what the correct threshold should be. If this project were adapted
for production use inside an actual HR system, these thresholds would
benefit from review against real intervention outcomes over time.
Additionally, K-Means assumes roughly spherical, similarly sized clusters,
and while the Hierarchical clustering validation step increases confidence in
the four-segment structure, it does not eliminate every limitation of the
underlying algorithm choice.

## Possible Future Improvements

- Replace the fixed promotion-gap risk thresholds with a model that is
  periodically recalibrated against actual employee outcomes.
- Add a time-series component, since the current dataset is a single snapshot
  and cannot show whether an individual employee's risk is increasing or
  decreasing over time.
- Extend the clustering feature set to include compensation-related
  variables, which were deliberately excluded from the current clustering
  model to keep the segments focused purely on career-trajectory signals
  rather than pay.
- Add authentication and role-based access to the Streamlit dashboard before
  any real deployment, since it currently displays individually identifiable
  employee data with no access controls.

## Dataset Source

This project uses an HR employee-attrition style dataset structured around
demographic, compensation, and tenure-related fields for a fictional company
context. "Palo Alto Networks" is used here as the case-study framing for this
internship project, not as a claim about that company's actual internal data.
If you are adapting this project, note where your own dataset came from here.

## Acknowledgments

This project was completed as part of the Unified Mentor internship program.
