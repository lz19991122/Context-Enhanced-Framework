import openai
from typing import List

openai.api_key = "your-api-key"

def read_file_lines(filename: str) -> List[str]:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found {filename}")
        return []
    except Exception as e:
        print(f"Read the file {filename} error occurred when: {str(e)}")
        return []

def revise_radiology_report(base_report: str, diagnostic_finding: str) -> str:
    prompt = f"""
    As a medical report editor, modify this radiology report to incorporate the new finding:

    {base_report}

    New finding to incorporate: {diagnostic_finding}

    Rules:
    • Only modify what's necessary to reflect the new finding
    • Keep the original format and style
    • Start directly with the report content
    • Do not add any labels, prefixes or explanations
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical report editor. Provide only the report content without any labels or prefixes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while calling the API: {str(e)}")
        return ""
def process_reports(base_reports_file: str, findings_file: str, output_file: str):
    base_reports = read_file_lines(base_reports_file)
    findings = read_file_lines(findings_file)

    if not base_reports or not findings:
        print("Input file is empty or unreadable.")
        return
    if len(base_reports) != len(findings):
        print(f"Warning: number of base reports ({len(base_reports)}) vs. number of diagnostic findings ({len(findings)}) mismatch.")
        process_length = min(len(base_reports), len(findings))
    else:
        process_length = len(base_reports)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(process_length):
                print(f"Report being processed {i+1}/{process_length}")
                revised_report = revise_radiology_report(base_reports[i], findings[i])
                if revised_report:
                    f.write(revised_report + '\n')
                    print(f"Report {i+1} processing completed")
                else:
                    print(f"Report {i+1} processing failure")
    except Exception as e:
        print(f"An error occurred while writing to the output file: {str(e)}")

def main():
    base_reports_file = "base_report.txt"
    findings_file = "diagnostic_finding.txt"
    output_file = "revised_reports.txt"

    process_reports(base_reports_file, findings_file, output_file)
    print("The revised reports are in the revised_reports.txt file.")

if __name__ == "__main__":
    main()
