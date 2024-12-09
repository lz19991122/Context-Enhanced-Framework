# Context-Enhanced-Framework-for-Medical-Image-Report-Generation-Using-Multimodal-Contexts

We have provided you with examples in the documentation so that you can simply run the code to visualize the process and results.

**Configuration**

The code requires an OpenAI API key:

```python
openai.api_key = "your-api-key"
```

**File Structure**



**Input files:**

base_report.txt: Contains the original radiology reports (B)

diagnostic_finding.txt: Contains new findings to be incorporated (D)

**Output file:**

revised_reports.txt: Output file for the processed reports

**Corresponding to the paper**

Corresponding to the paper, (B) in the paper is in base_report.txt, (D) is in diagnostic_finding.txt, and (P) is in the code.

We show the (P) here:

```python
You are a medical report editor. Provide only the report content without any labels or prefixes. As a medical report editor, modify this radiology report to incorporate the new finding:
    {B}
    New finding to incorporate: {D}
    Rules:
    • Only modify what's necessary to reflect the new finding
    • Keep the original format and style
    • Start directly with the report content
    • Do not add any labels, prefixes or explanations
```
    
**Usage**

Run the processor:

```bash
python medical_report_processor.py
```

