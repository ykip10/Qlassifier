This is a VCE exam question classifier based on NLP techniques. We use a transformers model to classify exam questions based on predetermined topics given by VCAA [study designs](https://www.vcaa.vic.edu.au/curriculum/vce-curriculum/vce-study-designs/vce-study-designs). We also build an end-to-end data extraction & processing pipeline to streamline predictions. Currently in development. 

## Demo
You can demo the application by going to [the FastAPI docs](https://qlassifier-app-233789415416.asia-southeast1.run.app/docs) and following these instructions:

- Uncollapse the /run_instructor endpoint and click "Try Out."
- Enter a subject name & supply a past VCAA exam for that subject
- Click run
- Results will show as list of 3-tuples suggesting the top 3 topics for each question from the exam, with associated confidence scores. 

# How to use Command Line Interface (CLI)
Make sure we're in the root directory.
```
cd path/to/repo
``` 

## Extracting VCAA Material
To download a subject's study design + reports and exams from the years year1,year2,year3:
```
python3 -m src.loader.material_collector subject_name year1,year2,year3
```
They will be saved into /data.

## Parsing Material (.docx / .pdf)

Material can be manually parsed with visualise output using
```
python3 -m src.parsing.visualise_parsing path_to_document
```
or with an optional additional ```--show-report``` flag if the material being parsed is a VCAA report. 

## Running Full Pipeline
We can run the full pipeline with 
```
python3 -m src.run_pipeline subject_name year [model]
```
If we want to download a VCAA past exam to run the model on, or
```
python3 -m src.run_pipeline subject_name exam_path [model] 
```
If we want to parse a custom exam at exam_path (which may or may not be a VCAA exam, although it should be examining a VCE subject).

where ```model``` is:
  - ```tf-idf``` 
  - ```instructor```
  
More options will become available.

# Notebooks

## Insights

```notebooks/generic_insights/``` contains simple statistical analyses on data extracted from parsed documents. 
In ```reports_data_analysis.ipynb```, we analyse the distribution of MCQ question answers in various subjects, 
as well as formulating an aggregate difficulty metric for each subject's MCQ sections, then ranking these.

## Model Exploration

```notebooks/modelling/``` contains simple explorations and early-model evaluations.  

# Roadmap:

- Document collection and parsing (completed) 
- Data analysis and visualisations (completed)
- Modelling (elementary approach completed, much room for improvement)
- Deployment (API deployed to Google Cloud)

# Current limitations: 

- Parsing does not recover LaTeX code. 
- Cannot parse from scanned PDFs. Text must be embedded. This means 2024 VCAA exams are unparsable. 
- Subjects with complex section layouts cannot be parsed reliably. 
