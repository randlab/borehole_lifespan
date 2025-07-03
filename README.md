## Borehole lifespan analysis
Metadata for Supplementary Documents – Borehole Lifespan Analysis

Description:

This supplementary material provides the data and code used for the analysis presented in the article:

Global insights into lifespan of water boreholes using survival analysis methods

Manon Trottet1*, Philippe Renard1, François Bertone2

1 Centre of Hydrogeology and Geothermics, University of Neuchâtel, 11 Rue Emile Argand, Neuchâtel, Switzerland
2 The World Bank, Washington, DC, USA 
* email : manon.trottet@unine.ch


# Contents:
1. **Dataset (ESM1.csv)**  
    - Contains borehole information. The variables include:
    - Last update year (YYYY)  
    - Borehole ID
    - Construction year (YYYY)
    - Nominal pumping rate (m3/s)
    - Last pumping rate (m3/s)
    - Year of decommissioning (NA for boreholes in operation, 9999 for unknown)

    **Data anonymization**:
    To preserve the confidentiality of the dataset—given that some of the borehole data are not publicly available—the entire dataset was anonymized. Of the original 2,128 borehole records, only 1,599 were retained. All information that could allow for the identification of borehole locations was removed, including region, country, and utility names. Borehole identifiers were anonymized and replaced with randomly assigned numbers from 1 to 1,599. To further prevent re-identification based on borehole characteristics, random noise was introduced to the borehole depth values (ranging from -1 to +1 meter), and the resulting depths were rounded. When available, initial and last known production rates (in m³/s) were rounded to four decimal places, introducing additional noise from the original values which were originally expressed in different units such as m³/year, m³/day, m³/hour, or gpm.


2. **Jupyter Notebook (ExampleAnalysis.ipynb)**  
    - Shows the survival analysis step by step, including Kaplan-Meier estimation, Gompertz model fitting, and RMST calculation.  
  

3. **Python code (BoreholesLifespan.py)**  
    - Contains the functions used in the notebook.


License:
Please cite the associated article if using this material.

Contact:
For any questions, please contact the corresponding author. 