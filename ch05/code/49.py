from google import genai
import sudachipy
client = genai.Client()

content = """
以下の文章のトークン数を数えてください。

「吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。
第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。
のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。
これが人間の飲む煙草というものである事はようやくこの頃知った。」

"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[{"role": "user", "parts": [{"text": content}]}],
) 

print("トークン数(AI)：", response.text)

# sudachipyを使ってトークン数を数える
tokenizer_obj = sudachipy.Dictionary().create()
mode = sudachipy.Tokenizer.SplitMode.C
tokens = tokenizer_obj.tokenize(content, mode)
token_count = len(tokens)
print("トークン数(sudachipy)：", token_count)

