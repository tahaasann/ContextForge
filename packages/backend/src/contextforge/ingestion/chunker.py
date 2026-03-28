from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from contextforge.ingestion.parser import ParsedDocument


@dataclass
class Chunk:
    text: str
    source_filename: str
    page_hint: int  # bu chunk hangi sayfadan geliyor (yaklaşık)
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Splitter bu sırayla bölmeyi dener:
        # önce çift newline (paragraf), sonra tek newline (satır),
        # sonra nokta/boşluk, son çare karakter karakter
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []

    # Sayfa sayfa işle — böylece hangi sayfadan geldiğini biliriz
    for page_idx, page_text in enumerate(doc.pages):
        if not page_text.strip():
            continue

        splits = splitter.split_text(page_text)

        for split_idx, split_text in enumerate(splits):
            chunks.append(
                Chunk(
                    text=split_text,
                    source_filename=doc.filename,
                    page_hint=page_idx + 1,  # 1-indexed, insan dostu
                    chunk_index=len(chunks),  # global index
                )
            )

    return chunks
