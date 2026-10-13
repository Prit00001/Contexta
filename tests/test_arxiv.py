from services.arxiv_client import ArxivClient

XML = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2005.11401v4</id><published>2020-05-22T00:00:00Z</published>
<title>Retrieval-Augmented
 Generation</title><summary> Large pre-trained models ... </summary>
<author><name>P. Lewis</name></author></entry><entry><id>x</id><title></title><summary>s</summary></entry></feed>"""


def test_parse_and_query():
    papers = ArxivClient.parse(XML)
    assert len(papers) == 1
    p = papers[0]
    assert p["url"] == "https://arxiv.org/abs/2005.11401v4" and p["id"] == "arxiv_2005.11401"
    assert p["title"] == "Retrieval- Augmented Generation".replace("- ", "-") or "Retrieval-Augmented" in p["title"]
    assert ArxivClient.build_query("What is RAG?") == "all:rag"
    assert " OR " in ArxivClient.build_query("how does retrieval augmented generation work")
