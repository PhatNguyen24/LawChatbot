from pdf2image import convert_from_path
import pytesseract

# Đường dẫn đến file PDF bạn muốn quét
pdf_file = 'data1/Tiền tệ ngân hàng và thị trường tài chính.pdf'

# Chuyển đổi trang thứ 6 của file PDF thành danh sách các hình ảnh
pages = convert_from_path(pdf_file, first_page=6, last_page=6)

# Thiết lập ngôn ngữ cho Tesseract OCR (nếu cần)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Đường dẫn đến tesseract trên macOS

# Mở tệp tin out.txt để ghi nội dung
with open('out.txt', 'w', encoding='utf-8') as file:
    # Duyệt qua từng trang hình ảnh và quét bằng pytesseract
    for i, page in enumerate(pages):
        # Quét hình ảnh và lưu nội dung vào tệp tin
        text = pytesseract.image_to_string(page)
        file.write(f"Content of page {i+1}:\n")
        file.write(text + '\n\n')
