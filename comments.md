# Comments

## 源码结构图

```plaintext

sglang/
    - launch_server.py      # 1. 启动服务器的脚本
    - srt/ (sglang runtime)
        - server_args.py    # 2. 服务器参数
        - entrypoint/
            - http_server.py  # 3. http服务器
            - engine.py       # 4. SRT引擎

```


## SRT 服务器

SRT服务器由 **HTTP服务器** 和 **SRT引擎** 组成

— HTTP服务器：FastAPI服务器，负责将请求路由到 SRT引擎。
- SRT引擎
  - TokenizerManager：tokenize请求并将其发送给调度程序。
  - Scheduler（子进程）：接收来自Tokenizer Manager的请求，调度batches，转发它们，并将输出tokens发送到Detokenizer Manager。
  - DetokenizerManager (子进程): 将输出tokens去detokenize，并将结果发送回TokenizerManager。

HTTP服务器、Engine和TokenizerManager都在主进程中运行
进程间通信通过IPC（每个进程使用不同的端口）通过ZMQ库完成
