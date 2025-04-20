# turns pdf into image

import fitz  # PyMuPDF

# Open the PDF
pdf = fitz.open("ltimindtree_annual_report.pdf")
page = pdf[1]  # Page 2 (index 1)

# Step 1: Inspect path_data from get_drawings()
path_data = page.get_drawings()
print(f"Total drawings extracted: {len(path_data)}")
for idx, item in enumerate(path_data, 1):
    print(f"\nDrawing {idx}:")
    print(f"  Item keys: {list(item.keys())}")
    path_items = item.get('items', [])
    print(f"  Number of path items: {len(path_items)}")
    for sub_item in path_items:
        print(f"    Sub-item type: {sub_item[0]}")
        if sub_item[0] == 'pa':  # Path item
            path_obj = sub_item[1]
            try:
                d = path_obj.get_d()
                print(f"    Path data: {d[:100]}...")  # Truncate for readability
            except Exception as e:
                print(f"    Error getting path data: {e}")
        else:
            print(f"    Non-path item: {sub_item}")

# Step 2: Try rendering the page as an SVG
try:
    svg_content = page.get_svg_image(matrix=fitz.Matrix(1, 1))
    with open("page2_full.svg", "w") as f:
        f.write(svg_content)
    print("Saved entire page as page2_full.svg")
except Exception as e:
    print(f"Error generating SVG for page: {e}")

# Step 3: Save the page as a PNG for visual inspection
pix = page.get_pixmap(dpi=300)
pix.save("page2.png")
print("Saved page as page2.png")

# Close the PDF
pdf.close()