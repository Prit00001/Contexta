from core.chunking import Block, chunk_blocks


def test_does_not_split_mid_sentence():
    text = "First sentence here. Second sentence here. Third one."
    chunks = chunk_blocks([Block(text=text)], max_chars=30)
    assert all(c.content.rstrip().endswith((".", "!", "?")) for c in chunks)
    assert " ".join(c.content for c in chunks).replace("  ", " ") == text


def test_carries_page_and_heading():
    chunks = chunk_blocks([Block(text="Some content.", page_number=3, section_heading="Intro")])
    assert chunks[0].page_number == 3
    assert chunks[0].section_heading == "Intro"


def test_chunk_index_increments_across_blocks():
    chunks = chunk_blocks([Block(text="A."), Block(text="B.")])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
