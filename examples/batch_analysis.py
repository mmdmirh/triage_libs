import os
import sys
import glob
import pandas as pd

# Ensure we can import triage_libs from parent directory if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from triage_libs import data_loader
from triage_libs.waittime_analysis import WaitTimeAnalyzer

def main():
    # Setup paths
    input_dir = "/Users/mohamad/Desktop/triage/csv_files"
    output_reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    output_plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    
    # Create output directories if they don't exist
    os.makedirs(output_reports_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # Initialize tools
    loader = data_loader.DataLoader()
    waititme_analyzer = WaitTimeAnalyzer()
    
    # List files
    csv_pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(csv_pattern)
    
    print(f"Found {len(files)} CSV files in {input_dir}")
    
    for file_path in files:
        try:
            filename_with_ext = os.path.basename(file_path)
            filename = os.path.splitext(filename_with_ext)[0]
            
            print(f"\nProcessing {filename}...")
            
            # 1. Read Data
            data = loader.read_file(file_path)
            
            # 2. Run Pipeline
            df_processed, summary = waititme_analyzer.run_analysis_pipeline(data)
            
            # 3. Generate Yearly Report (Start Month 9)
            yearly_breach_report = waititme_analyzer.generate_yearly_breach_report(df_processed, start_month=9)
            
            # Setup Paths
            plot_path_yearly = os.path.join(output_plots_dir, f"{filename}_yearly_trend.png")
            plot_path_hist = os.path.join(output_plots_dir, f"{filename}_breach_hist.png")
            report_excel_path = os.path.join(output_reports_dir, f"{filename}.xlsx")
            
            # 4. Plot Yearly Trend & Save
            waititme_analyzer.plot_yearly_breach_trend(
                yearly_breach_report,
                y_col="Breach_Count",
                clinic_name=filename,
                save_path=plot_path_yearly
            )
            
            # 5. Save to Excel (Multiple Sheets)
            # Sheet 1: Yearly Breach Report
            waititme_analyzer.save_to_excel(yearly_breach_report, report_excel_path, sheet_name="yearly_breach_report")
            
            # Sheet 2: Processed Data
            waititme_analyzer.save_to_excel(df_processed, report_excel_path, sheet_name=f"{filename} processed")
            
            # Sheet 3: Summary (Note: Calling save_to_excel on same file appends due to our lib update)
            waititme_analyzer.save_to_excel(summary, report_excel_path, sheet_name="summary")
            
            # 6. Plot Histogram & Save
            waititme_analyzer.plot_breach_margin_histogram(df_processed, save_path=plot_path_hist)
            
            # 7. Insert Images into Excel
            # Insert Yearly Plot into 'yearly_breach_report' sheet
            waititme_analyzer.insert_image_to_excel(report_excel_path, "yearly_breach_report", plot_path_yearly, "G2")
            
            # Insert Histogram into 'summary' sheet (or processed data)
            waititme_analyzer.insert_image_to_excel(report_excel_path, "summary", plot_path_hist, "G2")
            
            print(f"Completed processing for {filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
