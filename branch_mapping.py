import pandas as pd
from openpyxl import load_workbook

def append_to_excel(file_path, data, sheet_name='Sheet1'):
    
    # Load the existing worksheet
    df_old = pd.read_excel(file_path)
    
    # Convert the new data to a DataFrame
    df = pd.DataFrame(data)

    # Append the new data to the existing worksheet
    startrow = df_old.shape[0] + 1
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=True)

import pymupdf
# Open some document, for example a PDF (could also be EPUB, XPS, etc.)
doc = pymupdf.open(r"C:\Users\OEAGS58\Desktop\list-of-serviceable-in-codes.pdf")
# https://www.indusind.com/content/dam/indusind-corporate/Other/list-of-serviceable-in-codes.pdf

from openpyxl import load_workbook

Row = 0
for page in doc:
    tabs = page.find_tables()[0].to_pandas()
    if Row == 0:
        # Load existing workbook
        tabs.to_excel(r'C:\Users\OEAGS58\Desktop\STATE_MAPPING.xlsx', index = False)
    else:
        append_to_excel(r'C:\Users\OEAGS58\Desktop\STATE_MAPPING.xlsx', tabs)
    Row = Row + 1
