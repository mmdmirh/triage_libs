import pandas as pd
import os

import getpass
import io
import zipfile
try:
    import xlrd
except ImportError:
    xlrd = None
try:
    import msoffcrypto
except ImportError:
    msoffcrypto = None

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
                return pd.read_excel(file_path)
            except (zipfile.BadZipFile, xlrd.XLRDError if xlrd else Exception) as e:
                # If msoffcrypto is available, try to decrypt
                if msoffcrypto:
                    print(f"\nEncrypted file detected: {file_path}")
                    try:
                        password = getpass.getpass(prompt="Enter password: ")
                        decrypted_workbook = io.BytesIO()
                        
                        with open(file_path, "rb") as f:
                            office_file = msoffcrypto.OfficeFile(f)
                            office_file.load_key(password=password)
                            office_file.decrypt(decrypted_workbook)
                        
                        print("Decryption successful. Loading data...")
                        return pd.read_excel(decrypted_workbook)
                    except Exception as decrypt_error:
                        print(f"Failed to decrypt or load file: {decrypt_error}")
                        raise e # Raise original error if decryption fails
                else:
                    print("File might be encrypted but msoffcrypto-tool is not installed.")
                    raise e
            except Exception:
                 # Fallback/Retry if needed
                 if ext == '.xlsx':
                     return pd.read_excel(file_path, engine='openpyxl')
                 else:
                     return pd.read_excel(file_path, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
