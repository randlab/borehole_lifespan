# Borehole lifespan analysis

This archive contains the data and an example of jupyter notebook in python, that can be used to analyse the lifespan of water supply boreholes in the world following the methodology and results published in the joint paper cited below.

July 2025 - Manon Trottet - manon.trottet@unine.ch

### Reference

Trottet, M., P. Renard and F. Bertone, 2025,  Global insights into lifespan of water boreholes using survival analysis methods, Hydrogeology Journal, in review.

## Content

1. **Dataset (ESM1.xlsx)**

   - Contains borehole information. The variables include:
   - Last update year (YYYY)
   - Borehole ID
   - Construction year (YYYY)
   - Nominal pumping rate (m3/s)
   - Last pumping rate (m3/s)
   - Year of decommissioning (NA for boreholes in operation, 9999 for unknown)

   **Data anonymization**:
   To preserve the confidentiality of the dataset—given that some of the borehole data are not publicly available—the entire dataset was anonymized. Of the original 2,128 borehole records, only 1,599 were retained. All information that could allow for the identification of borehole locations was removed, including region, country, and utility names. Borehole identifiers were anonymized and replaced with randomly assigned numbers from 1 to 1,599. To further prevent re-identification based on borehole characteristics, random noise was introduced to the borehole depth values (ranging from -1 to +1 meter), and the resulting depths were rounded. When available, initial and last known production rates (in m³/s) were rounded to four decimal places, introducing additional noise from the original values which were originally expressed in different units such as m³/year, m³/day, m³/hour, or gpm.
2. **Jupyter Notebook (ESM2.ipynb)**

   - Shows the survival analysis step by step, including Kaplan-Meier estimation, Gompertz model fitting, and RMST calculation.
3. **Python code (ESM3.py)**

   - Contains the python functions used in the notebook.

## License

The code and data provided in this archive are provided without any warrantee.

When using either the code, or the data, you must cite the joint article (see references above)
