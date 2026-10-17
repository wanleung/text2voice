from app.text import clean_markup, detect_language, prepare, split_paragraphs


def test_markdown_is_stripped():
    md = "# Title\n\nSee [the docs](https://x.y) and **bold** `code`.\n\n```py\nprint(1)\n```\n\n![img](a.png)"
    assert clean_markup(md).split() == ["Title", "See", "the", "docs", "and", "bold", "code."]


def test_headings_and_lists_stay_separate_paragraphs():
    md = "Intro text.\n\n## 標題\n\n正文。\n\n- one\n- two\n"
    assert split_paragraphs(clean_markup(md)) == ["Intro text.", "標題。", "正文。", "one. two."]


def test_detect_language():
    assert detect_language("Hello there, how are you?") == "en"
    assert detect_language("今日天氣好好。") == "yue"
    assert detect_language("今日天氣好好。", chinese_default="zh") == "zh"
    assert detect_language("Apple 今日發佈咗新嘅 iPhone") == "yue"


def test_mixed_paragraph_switches_language_per_sentence():
    segs = prepare('佢話：「好正。」CEO Tim Cook said: "Best ever." 佢仲會開新店。')
    assert [s.language for s in segs] == ["yue", "en", "yue"]
    assert [s.paragraph_end for s in segs] == [False, False, True]


def test_same_language_sentences_are_merged():
    segs = prepare("Mr. Smith paid $15.50 today. It was fine.\n\nSecond paragraph.")
    assert [(s.text, s.language) for s in segs] == [
        ("Mr. Smith paid $15.50 today. It was fine.", "en"), ("Second paragraph.", "en")]


def test_forced_language_keeps_whole_paragraphs():
    segs = prepare("Hello. 你好。", language="yue")
    assert [(s.text, s.language) for s in segs] == [("Hello. 你好。", "yue")]
