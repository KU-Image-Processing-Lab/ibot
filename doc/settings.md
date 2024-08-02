
# 環境構築

```bash
$ cd docker/{docker-compose.ymlがあるディレクトリ}
$ docker-compose up -d
```

### 環境変数の設定

自作モジュールをimportするために、PYTHONPATHを設定してください。

直接環境変数を設定するには以下のコマンドを実行してください。この設定はそのシェルを消さない限り有効になります.

```bash
$ export PYTHONPATH=<リポジトリのルートディレクトリ>:$PYTHONPATH
```

いちいち設定が面倒な場合は、Bashを使用ている場合は`~/.bashrc`に、Zshを使用している場合は`~/.zshrc`に以下の行を追加してください。

```bash
$ export PYTHONPATH=<リポジトリのルートディレクトリ>:$PYTHONPATH >> ~/.zshrc

$ source ~/.bashrc
or
$ source ~/.zshrc
```

一度だけ環境変数を有効にする場合は、Pythonコマンドの前に環境変数を設定してください。

```bash
$ PYTHONPATH=<リポジトリのルートディレクトリ> python3 <スクリプト名>
```

これで`lib`や`utils`などのディレクトリをimportすることができます。


