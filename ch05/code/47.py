from google import genai

client = genai.Client()

contents = """
           あなたは川柳専門家として
           以下の10の川柳に対して面白さを評価してください
           10段階の評価で、次のことを評価基準としてください

           ・ユーモアや皮肉、風刺が効いているか
           ・人間の心理や行動を鋭く捉えているか
           ・軽妙洒脱で爽快感があるか
           各川柳に対して、評価点数のみを数字で答えてください
           1.  大掃除 見て見ぬふりの 来年へ
           2.  実家では 誰もが子供に 逆戻り
           3.  おせち見て 飽きる体に 罪悪感
           4.  お年玉 渡す立場も 悪くねぇ
           5.  仕事始め 頭はまだね 夢の中
           6.  新年の 抱負を語る 妻の圧力
           7.  年末の 挨拶メール 返信渋る
           8.  箱根駅伝 なぜか涙腺 ゆるむ午後
           9.  正月太り サイズ戻せず 春を待つ
           10. SNS 新年の挨拶 誰とする
           """
           
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[{"role": "user", "parts": [{"text": contents}]}],
) 
print(response.text)