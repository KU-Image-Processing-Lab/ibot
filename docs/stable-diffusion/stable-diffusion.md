### Stable Diffusionの導入, 学習の仕方から画像生成まで

1. Stable Diffusionの導入 : 

   1.1 [stable diffusion web uiのインストールの仕方](https://cysec.ise.ritsumei.ac.jp/2022/10/17/%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%82%B5%E3%83%BC%E3%83%90%E3%81%AB-stable-diffusion-web-ui-%E3%82%92%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)
   
      上記のサイトの説明に従って実行してください. これはローカルにダウンロードする時の説明であるため, 仮想環境ではstable diffusion web ui dockerで検索してください.

   1.2 Stable Diffusionの実行
        
      カレントディレクトリをstable-diffusion-webuiまで移動し, 以下のコマンドを実行してください.

      ```bash
       $ bash webui.sh
      ```

**注意事項**

ここでは, 遭遇したエラーと対処法について, 簡単に記載します.

**・ライブラリのversionが古い**
   
→ bash webui.shで立ち上げた仮環環境内でライブラリのversionを変更してください. また, versionの変更によって, 他のライブラリにエラーが出る場合があります. 
その場合はversionを下げることを行ってください. **pythonのversionにも気をつけてください.**

**・プロキシによって, Stable Diffusionを立ち上げてgenerateボタンを押してもエラーが出る**

→ webui-user.shの13行目に以下のコマンドを入力してください.
   

   