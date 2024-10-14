# Context-Enhanced-Framework-for-Medical-Image-Report-Generation-Using-Multimodal-Contexts

For label extraction, please follow our steps.

1. Please make sure that you have preprocess the medical reports accurately.

2. For MIMIC-CXR dataset, use the script located at tools/report_extractor.py for preprocess.

3. For the IU Xray, please download and use the preprocessed file from this [Download](https://raw.githubusercontent.com/ZexinYan/Medical-Report-Generation/master/data/new_data/captions.json)

4. Get disease labels using [Download](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert)

5. Use tools/count_nounphrases.py to generate additional labels. The output will be a JSON file that can be integrated with datasets like MIMIC or IU X-ray.


As part of our ongoing commitment to this project, we will continue to refine and expand these steps to ensure usability.