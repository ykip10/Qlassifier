This is a VCE exam question classifier based on NLP techniques. We use a transformers model to classify exam questions based on predetermined topics given by VCAA [study designs](https://www.vcaa.vic.edu.au/curriculum/vce-curriculum/vce-study-designs/vce-study-designs). We also build an end-to-end data extraction & processing pipeline to fine-tune our model. Currently in development. 

To download a subject's study design + reports and exams from the years year1,year2,year3:
```
python3 -m src.loader.material_collector subject_name year1,year2,year3
```
They will be saved into /data.
Material can then be manually parsed with

```
python3 -m src.parsing.visualise_parsing path_to_document
```
or with an optional additional ```--show-report``` flag if the material being parsed is a VCAA report. 

So far, only manual parsing is supported.

Roadmap:

- Document collection and parsing (completed) 
- Data analysis and visualisations (partially completed)
- Modelling (not begun)
- Deployment (not begun)

Current limitations: 

- Parsing does not recover LaTeX code. 
- Cannot parse from scanned PDFs. Text must be embedded. This means 2024 VCAA exams are unparsable. 
- Parser is not guaranteed to work for all subjects for all years. In particular, it requires at most one MCQ section and at most one short-answer question section. 
  Subjects with complex section layouts cannot be parsed reliably. 