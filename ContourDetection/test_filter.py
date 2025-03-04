import re


ocr_text = """
S.C. PEPOU HETAII. S.R.L.
SIRADA BUCEGI. NA.19, FEZANIN
IN CENIRUL COMERCIAL HINVARAT MR.1
JXETUL QLUJ, MNICIPIU. CLUD NAPECA
Cod Identificare Fiscăla: RO31477663

Lel
1,000 buc x 4,99
30038635 SET 4 BUC. SUPORTURI 4.9 A
PLU: 30038635
Extră PLU: 2207430038667
TOTAL 4.99
TOTAL TVA 0.80
TVAA 19.002 0.80
CASH 5.0
Rest
CASH 0.01-
Trz.: 3080-2-448639
Cas.: 3104004

Va mul tumtm pentru cumearaturi!
Reincârcati telefaul în releaua Pepco!
Gâsiţi oferta de produse pe kWww.pepco.ro
Bunrile se pot returna în baza bonului
1n max 30 zile de 1a data adlzi tionării

tn orice maoazin PEPCO din tară.

Produsele din caleooria lenJerle nu se
returnează. Multunim pentru inţelegere!

EJIRZ: 00238
ID UNIC: 80002794512024120523162122320238
PF. 00238 TA 06/12/2024 _ 0RA:20-16-24
8000279458
8 ON FISCAL
"""

def filter_relevant_lines(text):
    lines = text.split("\n")
    relevant_lines = []
    capture_next_line = False

    for i, line in enumerate(lines):
        line = line.strip()

        if re.search(r"\d{1,3}[.,]?\d{0,2}.*(Kg|Buc|X|TOTAL)", line, re.IGNORECASE):
            relevant_lines.append(line)
            capture_next_line = True

        elif capture_next_line:
            relevant_lines.append(line)
            capture_next_line = False

    return "\n".join(relevant_lines)

filtered_text = filter_relevant_lines(ocr_text)

print("Filtered Text:\n", filtered_text)
