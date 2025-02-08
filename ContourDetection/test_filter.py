import re

def filter_relevant_lines(text):
    lines = text.split("\n")  # Split text into lines
    relevant_lines = []
    capture_next_line = False  # Flag to capture product name after quantity

    for i, line in enumerate(lines):
        line = line.strip()  # Remove extra spaces

        # If the line contains a price, quantity, or "TOTAL", keep it
        if re.search(r"\d{1,3}[.,]?\d{0,2}.*(Kg|Buc|X|TOTAL)", line, re.IGNORECASE):
            relevant_lines.append(line)
            capture_next_line = True  # Mark next line as product name

        elif capture_next_line:  # Capture product name (appears after quantity)
            relevant_lines.append(line)
            capture_next_line = False  # Reset flag

    return "\n".join(relevant_lines)  # Join lines back into text

# Example OCR output (unfiltered text)
ocr_text = """
MEGA IMAGE SRL
JUD. CLUJ, MUNICIPIUL CLUJ-NAPOCA
PIATA UNIRII, NR. 2Z. PARTER. AP. 9
Cod [aentificare Fiscala: R06719278

2]
1,000° BUC, x 2,99
CICC DARK WAFESH0G Zi o2l
1,000 S RLIGE I dFod
ALKA CIOC AL350G 14,29 A
O LT S
TOTAL TVA 210
VA A 19,00% 2,76
CARD 17,28

ING
(=140 LT TRANZACTIT:

ING 17,28
VR

o b x| F oW 2V WVl v ..

"""

# Run the filter function
filtered_text = filter_relevant_lines(ocr_text)

print("Filtered Text:\n", filtered_text)
