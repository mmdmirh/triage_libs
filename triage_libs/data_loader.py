import pandas as pd
import os

class DataLoader:
    def read_file(self, file_path):
        """
        Reads a file (CSV or Excel) and returns a pandas DataFrame.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            try:
                # pandas usually auto-detects, but if it fails we can try specifying engine
                # for xlsx, openpyxl is preferred. for xls, xlrd is needed.
                return pd.read_excel(file_path)
            except Exception:
                 # Fallback/Retry if needed, but standard read_excel should work if deps are there.
                 # The error 'Can't find workbook...' suggests it might be trying to read xlsx as xls with xlrd.
                 # Let's try explicitly setting engine based on extension if general read fails?
                 # Actually, let's keep it simple first. 'read_excel' is smart.
                 # Providing engine='openpyxl' for .xlsx might help if it defaults to xlrd incorrectly.
                 if ext == '.xlsx':
                     return pd.read_excel(file_path, engine='openpyxl')
                 else:
                     return pd.read_excel(file_path, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
