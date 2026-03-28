from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class ParsedDocument:
    # dataclass: __init__, __repr__ otomatik gelir.
    # Pydantic kadar ağır değil - sadece veri taşıyıcı.
    filename: str
    pages: list[str]  # her sayfa ayrı string
    total_pages: int

    @property
    def full_text(self) -> str:
        # Sayfaları birleştir, araya çift newline koy.
        # Neden çift? Chunker bu sınırı bölme noktası olarak kullanacak.
        return "\n\n".join(page for page in self.pages if page.strip())


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    # BytesIO: disk'e yazmadan byte'ı dosya gibi okut.
    # Production'da geçici dosya oluşturmak hem yavaş hem güvenlik riski.
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Normalize et: fazla boşlukları temizle ama paragraf sınırlarını koru
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        pages.append(text)

    return ParsedDocument(
        filename=filename,
        pages=pages,
        total_pages=len(pages),
    )
