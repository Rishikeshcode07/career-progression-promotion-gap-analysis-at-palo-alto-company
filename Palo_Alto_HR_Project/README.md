# Palo Alto Networks — Career Progression & Promotion Gap Analysis (v3)

## What's new in this version
Steps 1-3 are now real Jupyter notebooks (.ipynb), not plain .py scripts.
Each notebook is broken into cells: a Markdown cell explaining WHAT the
next code does, WHY we're doing it, and WHAT RESULT to expect, followed by
a Code cell that runs it and displays the chart(s) directly underneath —
no need to dig through a charts/ folder. The notebooks have already been
executed once, so every chart is pre-rendered inside them; just open and
scroll, though you can also re-run them yourself (Cell > Run All).

## How to run
1. Make sure "Palo Alto Networks.csv" is in this same folder.
2. pip install -r requirements.txt
3. Launch Jupyter:  jupyter notebook   (or open the .ipynb files in
   VS Code / JupyterLab / Google Colab)
4. Open and run, in order:
     1_Data_Exploration.ipynb
     2_Feature_Engineering_Preprocessing.ipynb
     3_ML_Clustering.ipynb
5. Then launch the dashboard from a terminal in this folder:
     python -m streamlit run 4_app.py

## Files
- *.ipynb            -> the 3 analysis notebooks (cells + explanations + charts)
- 4_app.py            -> Streamlit dashboard (reads Processed_HR_Data.csv)
- Processed_HR_Data.csv, Cleaned_Engineered_HR_Data.csv, cluster_summary.csv
  -> pre-generated outputs, included so the dashboard works even before
     you run the notebooks yourself
- charts/             -> all 13 chart PNGs, also embedded inline in the notebooks
