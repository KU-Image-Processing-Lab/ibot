# モデルを学習・テストする方法

学習するためのソースコードは`./src`ディレクトリに格納されています。

```
src
  | mae
  | nakano
  | sam
  | transcaranet
  | transcaranet-finetuning
  | transcaranet-mtl
  | transfuse
  | transfuse-at
  | transfuse-finetuning
  | transfuse-mtl
  | transfuse-mtl-preprocessing
  | u-pranet
  | vae
```

例えば、TransFuseを使用したMTLの学習を行いたい場合、`./src/transfuse-mtl`ディレクトリに学習、テストそれぞれのファイルが格納されています。

```
transfuse-mtl
    | test_trans_nash.py
    | train_trans_nash.py
```

学習は基本的に直接`.py`を実行するのではなく、シェルスクリプトを使用して行います。

シェルスクリプトは`./scripts`ディレクトリに格納されています。

このシェルスクリプトを使用する意図は、学習時のパラメータを変更する際に、シェルスクリプトの引数を変更
するだけで学習を行えるようにするためと、あとは実験を再現しやすくするためです。

`./scripts`ディレクトリには以下のようなファイルが格納されています。

```
script
   | README.md
   | example.slack_notification.sh
   | example.transfuse_resnet10.sh
   | transcara_res2net_cv.sh
   | transfuse_ikeda.sh
   | transfuse_resnet10_const_param_d_cv.sh
   | transfuse_resnet10_cv.sh
   | transfuse_resnet10_preprocess.sh
   | transfuse_resnet10_preprocess_cv.sh
```

これを実行する際には、以下のようにbashコマンドを入力しましょう。

```bash
$ bash ./scripts/transfuse_resnet10_cv
```

ちなみに、このシェルスクリプトの中に、訓練およびテストの両方のコマンドを記述すると、いちいち実行する手間が省けるので便利です。

スクリプトの書き方は`./scripts`ディレクトリ内の`README.md`を参照してください。
