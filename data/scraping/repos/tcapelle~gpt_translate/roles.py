from textwrap import dedent

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def filter_dictionary(query, dictionary):
    "Filter out words from the query that are not in the dictionary"
    dictionary = dictionary.split("\n")
    filtered_dict = []
    for line in dictionary:
        dict_word = line.split(":")[0].lower()
        if dict_word in query.lower():
            filtered_dict.append(line)
    return "\n".join(filtered_dict)


system_template = """You are a translation assistant from {input_language} to {output_language}. Some rules to remember:
- Do not add extra blank lines.
- The results must be valid markdown
- It is important to maintain the accuracy of the contents but we don't want the output to read like it's been translated. So instead of translating word by word, prioritize naturalness and ease of communication.
- In code blocks, just translate the comments and leave the code as is.
Here is the translation dictionary for domain specific words:
- Use the dictionary where you see appropriate.
<Dictionary start>
{input_language}: {output_language}
{dictionary}
<End of Dictionary>
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = """Here is a chunk of Markdown text to translate. Return the translated text only, without adding anything else. 
<Text>
{text}
<End of Text>"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

DICTIONARIES = dict(
    ja="""\
access: アクセス
accuracy plot: 精度図
address: アドレス
alias: エイリアス
analysis: 分析
API key: APIキー
application:アプリケーション
architecture: アーキテクチャー
arg: ARG
argument: 引数
artifact: アーティファクト
assignment: 課題
autonomous vehicle: 自動運転車
AV model: AVモデル
backup: バックアップ
baseline: ベースライン
Bayesian search: ベイズ探索
behavior :振る舞い
bias: バイアス
blog: ブログ
bucket: バケット
business context: ビジネスコンテキスト
chatbot: チャットボット
checkpoint: チェックポイント
cloud: クラウド
cluster:クラスター
Colab notebook: Colabノートブック
computer vision: コンピュータビジョン
configuration: 設定
convolutional block: 畳み込みブロック
course: コース
customer case study: 顧客ケーススタディ
cutting-edge: 最先端の
dashboard: ダッシュボード
data: データ
data leakage: データ漏洩
data obfuscation: データ難読化
data visualization: データ可視化
dataset: データセット
dataset-agnostic: データセットに依存しない
deep learning: ディープラーニング
demo: デモ
deployment:	展開
directory: ディレクトリー
docker container: dockerコンテナ
ecosystem: エコシステム
edge case: 極端なケース
end-to-end: エンドツーエンド
environment: 環境
epoch: エポック
experiment: 実験
fine-tune: 微調整
forward pass: forwardパス
ground truth: 正解
guide: ガイド
hook: フック
host flag: ホストフラグ
Hugging Face Transformer: Hugging Faceトランスフォーマー
hyperparameter: ハイパーパラメーター
hyperparameter sweep: ハイパーパラメーター探索
hyperparameter tuning: ハイパーパラメータチューニング
infrastructure: インフラストラクチャー
key: キー
library: ライブラリ
line of code: (抜き出し方が良くないが）"couple lines of code" で 「コード2行」となる
lineage: 履歴
local minima: 局所的最小値
log: ログ
machine learning: 機械学習
machine learning practitioner: 機械学習開発者
metadata: メタデータ
method: メソッド
metrics: メトリクス
ML practitioner: MLエンジニア
model: モデル
model evolution: モデルの進化
model lineage: モデルの履歴
model management: モデル管理
model registry: モデルレジストリ
model training: モデルトレーニング
neural network: ニューラルネットワーク
noising: ノイジング
notebook: ノートブック
object: オブジェクト
on-prem: オンプレミス
Optimizer: オプティマイザー
orchestration: オーケストレーション
overfitting: 過学習
pipeline: 開発フロー
platform: プラットフォーム
population based training: 集団的学習
precision-recall curve: PR曲線
pre-trained: 学習済み
private cloud: プライベートクラウド
process: プロセス
processing: 処理する
production: プロダクション
project: プロジェクト
Quickstart: クイックスタート
recommender system: 推薦システム
reinforcement learning: 強化学習
report: レポート
reproducibility:再現性
result: 結果
run: run
runs: runs
SaaS: SaaS
script: スクリプト
semantic segmentation: セマンティックセグメンテーション
sentiment analysis: センチメント分析
server: サーバー
state assignments: 状態割り当て
subset: サブセット
support team:  サポートチーム
sweep: スイープ
sweep agent: スイープエージェント
sweep configuration: スイープ構成
sweep server: スイープサーバー
system of record: 記録システム
test set: テストセット
text-to-image: text-to-image
time series: 時系列
tool: ツール
track: トラッキング
tracked hours: 追跡時間
training: トレーニング
training data: トレーニングデータ
training script: トレーニングスクリプト
trial: 試験
tune: チューニングする
use case: ユースケース
user: ユーザー
validation accuracy: 検証精度
version: バージョン
versioning: バージョン管理
W&B Fully Connected: W&B Fully Connected
wandb library: wandbライブラリ
Weave expression: Weave式
""",
    es="""\
accuracy plot: gráfica de precisión
address: dirección
alias: alias
API key: clave API
application: aplicación
arg: ARG
argument: argumento
artifact: artefacto
AV model: modelo AV
backup: respaldo
baseline: referencia
Bayesian search: búsqueda bayesiana
bias: sesgo
blog: blog
bucket: bucket
business context: contexto de negocio
chatbot: chatbot
checkpoint: checkpoint
cloud: nube
cluster: cluster
Colab notebook: Colab Notebook
computer vision: visión por computadora
configuration: configuración
convolutional block: bloque convolucional
course: curso
customer case study: caso de estudio de cliente
cutting-edge: de última generación
dashboard: panel de control
data: datos
data leakage: filtración de datos
data obfuscation: ofuscación de datos
data visualization: visualización de datos
dataset: dataset
dataset-agnostic: dataset-agnóstico
deep learning: aprendizaje profundo
demo: demo
deployment: despliegue
directory: directorio
docker container: contenedor docker
ecosystem: ecosistema
edge case: caso extremo
embedding: embedding
end-to-end: end-to-end
environment: entorno
epoch: epoch
experiment: experimento
fine-tune: calibración fina
forward pass: evaluación directa
ground truth: verdad de campo
guide: guía
hook: hook
host flag: host flag
Hugging Face Transformer: Hugging Face Transformer
hyperparameter: hiperparámetro
hyperparameter sweep: barrido de hiperparámetros
hyperparameter tuning: ajuste de hiperparámetros
infrastructure: infraestructura
key: clave
library: biblioteca
line of code: linea de código
lineage: linaje
local minima: mínimo local
log: registro
machine learning: aprendizaje automático
machine learning practitioner: profesional de aprendizaje automático
metadata: metadatos
method: método
metrics: métricas
ML practitioner: profesional de ML
model: modelo
model evolution: evolución del modelo
model lineage: linaje del modelo
model management: gestión del modelo
model registry: registro de modelos
model training: entrenamiento del modelo
neural network: red neuronal
noising: ruido
notebook: notebook
object: objeto
on-prem: on-prem
Optimizer: Optimizador
orchestration: orquestación
overfitting: sobrecalibración
pipeline: pipeline
platform: plataforma
population based training: entrenamiento basado en población
precision-recall curve: curva de precisión-recuperación
pre-trained: pre-entrenado
private cloud: nube privada
process: proceso
processing: procesamiento
production: producción
project: proyecto
Quickstart: inicio rápido
recommender system: sistema de recomendación
reinforcement learning: aprendizaje por refuerzo
report: reporte
reproducibility: reproducibilidad
result: resultado
run: run
runs: runs
SaaS: SaaS
script: script
semantic segmentation: segmentación semántica
sentiment analysis: análisis de sentimiento
server: servidor
state assignments: asignaciones de estado
subset: subconjunto
support team: equipo de soporte
sweep: barrido
sweep agent: agente de barrido
sweep configuration: configuración de barrido
sweep server: servidor de barrido
system of record: sistema de registro
test set: conjunto de prueba
text-to-image: text-to-image
time series: series de tiempo
tool: herramienta
track: seguimiento
tracked hours: horas de seguimiento
training: entrenamiento
training data: datos de entrenamiento
training script: script de entrenamiento
trial: prueba
tune: ajustar
use case: caso de uso
user: usuario
validation accuracy: precisión de validación
version: versión
versioning: versionamiento
W&B Fully Connected: W&B Fully Connected
wandb library: biblioteca wandb
Weave expression: expresión Weave
""",
)
