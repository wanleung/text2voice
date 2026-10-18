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


def test_article_extraction_dedupes_and_drops_code():
    from app.fetch import extract
    body = """<p>First paragraph of the article, long enough to count.</p>
<pre><code>Get-MgUser -All | Export-Csv users.csv</code></pre>
<p>Second paragraph mentions the 郵箱 setting in some detail here.</p>"""
    html = f"""<html><head><title>My Title</title></head><body><article><h1>My Title</h1>{body}
<h1>My Title</h1>{body.replace("郵箱", "信箱")}</article></body></html>"""
    a = extract(html, "https://example.com/post")
    assert a.title == "My Title"
    assert a.text.count("First paragraph") == 1 and a.text.count("Second paragraph") == 1
    assert "Get-MgUser" not in a.text and a.text.count("My Title") == 1


def test_cantonese_model_routing():
    from app.engine import use_yue_model
    assert use_yue_model("今日天氣好熱，我哋去飲茶啦。", "yue")
    assert not use_yue_model("我哋會用Microsoft 365。", "yue")  # English words stay on the main model
    assert not use_yue_model("Hello there.", "en")
    assert not use_yue_model("今日天氣好熱。", "zh")


def test_divider_lines_and_stray_brackets_are_not_spoken():
    text = "標題\n==========\n第一段。\n～～～～\nIntro text.\n-----\n佢問：【點算好？\n】\n跟住講。"
    segments = prepare(text)
    assert [s.text for s in segments] == ["標題。", "第一段。", "Intro text.", "佢問：【點算好？】跟住講。"]
    assert all("=" not in s.text and "～" not in s.text for s in segments)


def test_progress_moves_within_a_long_segment():
    import numpy as np
    from app.engine import SegmentSynthesizer, report_chunks
    from app.text import Segment

    class Chunky(SegmentSynthesizer):
        sample_rate = 100

        def synthesize_segment(self, segment, voice, speed=1.0, on_chunk=None):
            for _ in report_chunks(segment.text.split("|"), on_chunk):
                pass
            return np.zeros(10, dtype=np.float32)

    calls = []
    Chunky().synthesize([Segment("aa|aa", "en"), Segment("b|b|b|b", "en")], None, pause_ms=0,
                        progress=lambda *a: calls.append(a))
    fractions = [f for _, _, f in calls]
    assert fractions == sorted(fractions) and fractions[-1] == 1
    assert len(calls) > 2 + 2  # updates inside segments, not only at their ends
    assert calls[-1][:2] == (2, 2)
