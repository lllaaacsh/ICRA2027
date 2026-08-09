# LARM 审稿意见 → 写作修改方案

> 依据：`ieeeconf/content/*.tex` 当前版本 + 实现代码
> (`LARM/mae/dino_tobo_hist.py`、`LARM/mae/train_dino_tobo.py`、
> `LARM/mae/train_tobo_main.py`、`LARM/metaworld_bc/model_wrapper.py`)。
> 本文档只提方案和草稿文字，**没有改任何 `content/*.tex`**。

---

## 状态（2026-08-09 更新）

**架构已按"memory 端非对称瓶颈"重做,论文的 method/motivation 已同步改完。**
本文档第 0 节记录的是**改之前**的状态,保留作为背景。

已完成:

| | 内容 |
|---|---|
| 代码 | `LARM/mae/larm_model.py` 新架构 · `engine_larm.py` 训练/验证 · `main_larm.py` 入口 · `train_larm.sh` |
| 论文 | `abstract` · `intro`(motivation + R3M 措辞 + contribution) · `related_work`(新增 latent action 小节 + Fig.2 caption) · `method`(全面重写 + 新增 III-D Deployment + 伪代码框) · `experiment`(k=96 澄清 + MT10 视觉输入) |
| 标记 | 所有改动用 `\rev{}` 标成黄色。定稿时把 `premeable.tex` 里的 `\rev` 改成 `\newcommand{\rev}[1]{#1}` 即可一键去掉 |

新架构 (与旧版的区别见第 0 节):

```
Query  Q : 1+N 个 learnable、content-free 的 token + 位置编码   ← 空间分辨率来自 query
Full   流: memory = [v'_cls ; v'_1..v'_N ; L']
Intent 流: memory = [v'_cls ;              L']                  ← 视觉只剩 1 个 token
两条流同一套权重、同样的算子序列, 唯一变量是能读到什么
输出 (B, 1+N, D): 索引 0 = 未来 CLS, 1..N = 未来 patch, 与 teacher 一一对齐
部署保留 intent 流 (student), full 流只在预训练期存在
```

冒烟测试已过: 两条流输出确实不同; intent 流对图像和语言都敏感 (瓶颈不是 0 带宽);
三个 loss 的梯度都能到达共享 decoder **和** fusion 模块 (这正是 AE 追问的耦合路径);
backbone 全程冻结。

还没做:

- [ ] 重新预训练 + 下游重测 (用户明确说放到后面)
- [ ] `metaworld_bc/model_wrapper.py` 还在用旧的 `LatentFramePredictorToBo`,
      要为新 checkpoint 加一个 policy 类
- [ ] 三张图重画 (第 9 节)
- [ ] 真机实验协议 / trial 数 (第 7 节, 需要用户提供事实)
- [ ] **真机遮挡 (R9-8)**: 相机确认是**腕部安装**,审稿人的读图没错。但答法取决于
      观测是在运动流程的哪个时刻拍的 (回 scan pose 拍一帧 / pick 与 place 各拍
      一帧 / 每步闭环) —— 用户待确认,确认前不动第 6.2 节那段文字
- [ ] 消融段与 MT10 的矛盾 (第 3 节, 需要用户拍板)

已否决: III-D 末尾的推理延迟段 (用户决定删掉;III-D 开头已经回答了 R9-6 的
执行频率问题,延迟数字是额外的)。

⚠️ 顺带发现的既有问题: `tables/franka.tex` 里有 20 个 `\cite` 键在 `mybib.bib`
里不存在 (`clip`/`r3m`/`mpi`/`voltron`/...),`tables/eval_set.tex` 还有一个
`zeng2021transporter`。这些不是本轮改出来的,但 Table I 很可能在渲染 `[?]`,
在 Overleaf 里确认一下。

---

## 0. 改之前的状态：代码实际做的事（保留作为背景）

我逐行核对了实现，有 **4 处论文与代码不符**。审稿人问的"看不懂"，大部分源头在这里，
不先定下来就没法改字。

| # | 论文怎么写 | 代码实际怎么做 | 出处 |
|---|---|---|---|
| **A** | 把 $F_{\text{full}}$ / $F_{\text{intent}}$ 写成两个模块（只在一句里提了 "shared weights"） | **同一套 decoder 权重跑两遍**。`x1, x2 = x.clone(), x.clone()`；每个 block 里 `blk(x1, y=y1)`（读 dense memory）vs `blk(x2, y=None)`（不读） | `dino_tobo_hist.py:200-215` |
| **B** | Eq.(4)：$F_{\text{intent}}(V'_{\text{cls}}, L')$，即"只输入 CLS token" | 两条流的 **query 序列完全相同**（CLS+196 patch）。瓶颈是**不给 cross-attention memory**，不是"只喂 CLS" | `dino_tobo_hist.py:83-89, 200-206` |
| **C** | §III-C + Fig.2 caption：部署时**丢掉 intent、保留 full** | 部署时用的是 **`pred_tobo`（= intent 流）**，`pred_full` 变量名带下划线表示丢弃。全仓库没有任何一处下游用 `pred_full` | `model_wrapper.py:438,442` |
| **D** | §III-C：把 **\[CLS\] residual** flatten 成 1D 向量 | 预测器输出 **只有 196 个 patch token，不含 CLS**（`decoder_pred(x[:, 1:, :])`）。$\Delta z$ 是逐 patch 的，pooled decoder 里是 mean-pool | `dino_tobo_hist.py:212-213`；`model_wrapper.py:442,458` |

另外三个可以直接拿去答复审稿人的事实：

- **DistilBERT 全程冻结**（`eval()` + `torch.no_grad()` + `requires_grad=False`），
  而且只取 `last_hidden_state[:,0,:]` → **语言侧只有 1 个 token**。
  ⇒ AE 猜的 "coupling via the text backbone" **不存在**；真正的耦合是共享的
  `lang_proj` + `fuse_blocks` + 共享 decoder 权重。这是个很干净的答复。
- **Teacher DINOv3 冻结；Student backbone 前 `frozen_epochs` 轮冻结，之后以 0.1× lr 解冻**
  （`train_tobo_main.py:305-318`）。论文完全没提这个两阶段，实现细节该补。
- **部署时预测器每个控制步都跑**（`act()` 每帧调 `forward()`），闭环。→ 直接回答 R9-6。

### ⚠️ C 这条特别重要

如果按代码写（**保留 intent 流、丢掉 full 流**），AE 的整个质疑**自动消失**，
方法逻辑立刻自洽：

```
L_full  : 训练共享权重，让"能读 dense memory"的这条前向准确建模场景演化 → 它是 teacher
L_intent: 训练同一套共享权重，让"读不到 memory"的前向也能预测 → 它是 student
L_cons  : full → intent 蒸馏，sg 打在 teacher 上 ← 这是标准做法，位置本来就对
部署    : 跑 student（intent 流）。teacher 只在预训练期存在。
```

AE 说"stop-gradient 应该打在 F_intent 上"，是因为论文说部署留 full；
一旦说清楚**留的是 intent**，sg 打在 full 上就是唯一正确的写法。
**我强烈建议按这个改**——它同时解决 AE-1、R9-5、R9-4 三条。

**但你必须先确认 Franka Kitchen 和真机那两套实验是不是也用的 intent 流。**
我这边只能看到 MT10 的代码路径（那里 100% 是 intent 流）。如果三套实验用的流不一致，
那是实质问题，不是写作问题，必须先统一。

---

## 1. AE-1：$F_{\text{intent}}$ 与 $F_{\text{full}}$ 的训练关系（最高优先级）

AE 的原话是 "difficult to assess the paper based on guesswork" —— 他要的不是解释，
是**不需要猜**。所以除了改字，一定要加一个**伪代码框**。这是性价比最高的一处修改。

### 1.1 §III-B 增加一段"参数共享"（放在 "We construct two parallel predictive streams" 那句的位置，替换它）

```latex
\noindent\textbf{One predictor, two access conditions.}
The two streams are not two networks. They are the \emph{same} predictor
$F_\theta$ evaluated twice under different access to spatial memory.
$F_\theta$ is a stack of decoder blocks, each composed of (i) a cross-attention
layer that reads a dense memory $M$, (ii) a self-attention layer, and (iii) an MLP;
when $M=\varnothing$ the cross-attention step is skipped and the block reduces to
self-attention and MLP. Writing $Q$ for the query tokens (the fused tokens plus
positional and temporal embeddings) and $V'$ for the language-grounded dense
features of Sec.~\ref{subsec:encoder}, we define
%
\begin{align}
    \hat z^{\text{full}}_{t+k} &= F_\theta\!\left(Q \mid M = V'\right), \\
    \hat z^{\text{int}}_{t+k}  &= F_\theta\!\left(Q \mid M = \varnothing\right).
\end{align}
%
The full-information pass can re-read fine-grained spatial evidence at every layer,
whereas the intent-aligned pass is cut off from it and must carry the future
through the language-grounded global context alone. Every parameter---the fusion
module of Sec.~\ref{subsec:encoder} and the predictor $F_\theta$---is shared
between the two passes; the information bottleneck is imposed on the
\emph{computation}, not on the parameters.
```

> 注意这段同时修好了 **B**：把"只输入 CLS"改成"不给 memory"。
> 如果你想保留"CLS 瓶颈"的说法（更好讲故事），那就必须去改代码让它名副实归，
> 否则一旦开源就露馅。我建议按代码写。

### 1.2 损失那段后面追加"梯度流向"（直接回应 AE 逐字质疑）

```latex
\noindent\textbf{Which parameters each term updates.}
Because $\theta$ is shared, all three terms update the same predictor, and their
roles are separated by the input condition and the stop-gradient rather than by
parameter partitioning. Eq.~\ref{eq:loss_full} supervises $F_\theta$ (and the
fusion module) through the memory-conditioned pass, and is the only term that
trains the cross-attention read-out. Eq.~\ref{eq:loss_intent} supervises the
\emph{same} parameters through the memory-free pass, forcing the fused global
context to be predictive on its own. Eq.~\ref{eq:consistency} distills the
memory-conditioned forecast into the memory-free one; the stop-gradient is placed
on the teacher side $\hat z^{\text{full}}$, so that the bottlenecked pass is pulled
towards the accurate forecast without the accurate forecast being dragged back
towards the deliberately impoverished one, which would otherwise collapse both
streams to their average. Consequently Eq.~\ref{eq:loss_intent} and
Eq.~\ref{eq:consistency} do shape the representation that is kept after
pre-training: they act on the shared fusion module and on $F_\theta$ itself, and
suppress any future component that cannot be recovered from task context alone.
The language backbone is frozen throughout and contributes no gradient path.
```

最后一句是专门写给 AE 看的（他提到 "coupling via the text backbone"）。

### 1.3 加一个伪代码框（**最重要的一处**）

放在 Eq.(6) 之后。IEEE 双栏里 `algorithm` 环境放单栏即可。

```latex
\begin{algorithm}[t]
\caption{LARM pre-training step (PyTorch-like)}
\label{alg:pretrain}
\begin{algorithmic}[1]
\State $V \gets \texttt{DINOv3}_{\text{student}}(I_t)$  \Comment{frozen for the first $E_f$ epochs}
\State $L \gets \texttt{Proj}(\texttt{DistilBERT}(c))$  \Comment{frozen, 1 token}
\State $V', L' \gets \texttt{BiCrossAttn}(V, L)$        \Comment{trainable, shared}
\State $Q \gets \texttt{Embed}(V') + \texttt{pos} + \texttt{time}$
\State $\hat z^{\text{full}} \gets F_\theta(Q, M{=}V')$ \Comment{same $\theta$}
\State $\hat z^{\text{int}}  \gets F_\theta(Q, M{=}\varnothing)$ \Comment{same $\theta$}
\State $z^{\text{gt}} \gets \texttt{DINOv3}_{\text{teacher}}(I_{t+k}).\texttt{detach()}$
\State $\mathcal{L} \gets \ell(\hat z^{\text{full}}, z^{\text{gt}}) + \ell(\hat z^{\text{int}}, z^{\text{gt}}) + \alpha\,\|\hat z^{\text{int}} - \texttt{sg}(\hat z^{\text{full}})\|^2$
\State backprop into $\{\texttt{BiCrossAttn}, \texttt{Proj}, \theta\}$ (and the student backbone after epoch $E_f$)
\end{algorithmic}
\end{algorithm}
```

### 1.4 §III-C 开头那句必须改（连带修 C）

原句：
> "we discard the intent-aligned regularizer and retain only $F_{\text{full}}$, now denoted as $z_{t+1}$"

建议（按代码）：

```latex
After pre-training, the memory-conditioned pass is no longer instantiated: it
exists only to provide the distillation target of Eq.~\ref{eq:consistency}.
What is deployed is the bottlenecked pass, which by construction predicts the
future from language-grounded task context and has been distilled to agree with
the full-information forecast. We denote its output $\hat z_{t+1}$ and compute a
latent residual ...
```

同时 **Fig.2 caption 里 "During deployment, the intent predictor is discarded" 这句要反过来写**。

---

## 2. R8-1：与 latent action training 的关系

建议在 Related Work 新开一个小节 `\subsection{Relation to latent action models}`，
或者接在 §II-B 后面。这是所有意见里**唯一需要新增独立段落**的一条，别塞进现有段落。

引用建议加：LAPO（Schmidt & Jiang, *Learning to Act without Actions*, ICLR'24）、
LAPA（*Latent Action Pretraining from Videos*, ICLR'25）、Genie（Bruce et al., ICML'24）。
先去核对审稿人说的 [1,2] 具体是哪两篇，把它们放在首位。

```latex
\subsection{Relation to latent action models}
A parallel line of work makes action-free video usable for policy pre-training by
\emph{inventing} an action space: an inverse dynamics model infers a latent action
$u_t$ from $(o_t, o_{t+k})$, a forward model is asked to reconstruct $o_{t+k}$ from
$(o_t, u_t)$, and a vector-quantised bottleneck on $u_t$ forces it to carry only the
controllable part of the transition~\cite{lapo,genie,lapa}. A policy is then
pre-trained to emit $u_t$, and a small action-labelled set aligns $u_t$ with real
commands.

LARM shares the premise---action-free video contains dynamics---but not the
mechanism. We do not learn an action variable at all. The quantity we extract,
$\Delta_z = \hat z_{t+1} - z_t$, lives in the \emph{observation} space of a frozen
perceptual encoder and is consumed by the policy as a conditioning feature, so no
latent-to-real action alignment stage is needed. Three consequences follow.
\emph{(i) Where the bottleneck comes from.} Latent action models must make $u_t$
low-capacity so that it captures controllable rather than passive change; this is
an unsupervised bottleneck and its meaning is only identified up to a
reparameterisation. LARM instead uses the \emph{language instruction} as the
bottleneck: the intent-aligned pass must forecast the future from task context
alone, so the retained transition is the one the instruction accounts for.
\emph{(ii) Cross-embodiment transfer.} A latent action inferred from human hands
in web video is defined relative to the human embodiment; transferring it to a
parallel-jaw gripper requires the alignment stage to absorb the mismatch. A state
residual has no embodiment attached---it is a direction in a perceptual space
shared by both---which is what allows LARM to pre-train on Kinetics and deploy on a
UR5 without any robot pre-training data.
\emph{(iii) How dynamics are learned without latent-action supervision.} The
supervision is the prediction task itself: the model must map $(o_t, c)$ to the
frozen-encoder embedding of $o_{t+k}$, so $\Delta_z$ is exactly the $k$-step
transition the instruction implies. LARM does not need to identify \emph{which
command} produced that transition, because the downstream policy is trained with
real action labels; the residual only has to say where the scene should go, not
what to send to the controller.

The trade-off is explicit: because $\Delta_z$ is not an action, it cannot be rolled
out for planning or used as a policy target on its own, which latent action models
can do. LARM is therefore complementary---it is a representation-side prior rather
than a substitute for the action space.
```

最后一段的"退让"很关键：审稿人问的是 "similarities, differences, **and advantages**"，
主动写清楚代价，比只吹优点更容易过。

---

## 3. R8-2：仿真太简单

MT10 已经加进去了，写作上还要做三件事，否则审稿人不一定认为你回应了：

1. **摘要**里点名两个 benchmark。当前 abstract 只说 "simulated and real-robot"，
   太虚。改成 `...on two simulation benchmarks (Franka Kitchen and Meta-World MT10)
   and seven real-robot tasks...`
2. **Intro 第三条 contribution** 现在还只写 Franka Kitchen（`intro.tex:95`），必须补 MT10。
3. **§IV-C 第一段**（`experiment.tex:80`）已经解释了两个 benchmark 的互补性，写得不错，
   建议再加一句显式对照："Franka Kitchen varies the sub-task within one scene; MT10
   varies the scene, the object and the motion primitive under a single policy."

⚠️ **同时必须处理消融段的矛盾**（`PAPER_HANDOFF.md` §5.1）。R8 要更多仿真，
新加的 MT10 恰恰把消融的结论推翻了：MT10 上 Predictor-Only (79.0) > Full (75.6)，
而正文写的是 "Predictor-Only provides only a marginal improvement… Full achieves a
dramatic +6.7%"。**下一轮审稿人一定会抓这个**。诚实且仍然有力的写法：

```latex
Two effects need to be separated: whether the policy sees the predicted future at
all, and whether that future is presented as an explicit residual. The first
effect is large and consistent. On MT10 every variant that receives predictive
information outperforms the encoder-only variant by 18--22 points
($>5\sigma$ given a per-arm standard error of $\approx 3.2$ points), which is the
evidence for our central claim that the predictor should be retained rather than
discarded after pre-training. The second effect is smaller and benchmark
dependent: on Franka Kitchen the residual parameterisation adds
$+5.7$ points over \textit{Predictor-Only}, whereas on MT10 the three predictive
arms are statistically indistinguishable from one another ($\le 1.3\sigma$). We
therefore claim only the first effect, and report the residual parameterisation as
the variant that is never worse and is markedly better on the longer-horizon
articulation tasks.
```

这样处理，你的核心 contribution（"保留预测器"）反而更稳，因为它有 6σ 证据；
而弱的那条主动降级，比被审稿人揪出来好得多。

---

## 4. R8-3：k=96 不合理

审稿人的误解是把 k 当成了**控制时域**。改法分三步：

### 4.1 §IV-A 补单位和角色（当前 `experiment.tex:23`）

```latex
Kinetics-400 is stored at 25--30\,fps, so $k=96$ corresponds to a gap of roughly
3--4 seconds of human activity. We stress that $k$ is a \emph{pre-training}
hyper-parameter over web video, not a control horizon: at deployment the predictor
is applied to the current robot observation at every control step and its output is
consumed as a direction (Sec.~\ref{subsec:policy_learning}), while the magnitude
and the time scale of $\Delta_z$ are absorbed by the action decoder, which is
trained by behavioural cloning against real actions.
```

### 4.2 给出选 96 的理由（不要只说 "as in [ToBo]"）

```latex
The gap has to be large enough that the transition is not dominated by encoding
noise. Frames one third of a second apart are near-identical under a frozen
DINOv3 encoder, so the prediction target degenerates towards an identity map and
the residual carries little task signal.
```

### 4.3 **强烈建议补一张小图/一行数字**（很便宜，说服力最高）

在 Kinetics 验证集上算 $\cos(z_t, z_{t+k})$ 随 $k$ 的曲线，标出 96 落在哪里。
一句话 + 一条曲线就能把"96 太长了"变成"96 是信噪比开始饱和的位置"。
如果还有算力，最好的答复是 **k ∈ {24, 48, 96, 144} 各预训练一版，报 MT10 平均成功率**。
这是审稿人真正想看的；只解释不给数，他可能会坚持。

---

## 5. R9-4 / R9-5 / R9-6：\[CLS\]、部署输出结构、执行频率

这三条其实是同一个洞：**论文从没写清楚"部署时那条 forward 到底算什么、多久算一次"**。
建议**新开一小节 §III-D "Deployment"**，一次性堵掉，而不是散在各处补。

### 5.1 先在 §III-A 把记号钉死（修 R9-4）

```latex
We write $V = [\,v_{\texttt{cls}}, v_1, \dots, v_N\,]$ for the visual tokens produced
by the DINOv3 backbone, where $v_{\texttt{cls}}$ is the visual class token and $N=196$
is the number of patches at $224\times224$ resolution. The language branch is a
frozen DistilBERT whose pooled sentence embedding is projected to a single token
$L$; we never use the text encoder's own special tokens as separate inputs, so
throughout the paper \texttt{[CLS]} refers exclusively to the \emph{visual} class
token, and $v'_{\texttt{cls}}$ denotes it after fusion.
```

> 顺带：语言侧只有 **1 个 token** 这件事论文完全没说，但 Eq.(2)
> $L' = \texttt{CrossAttn}(L, V', V') + L$ 读起来像是一个序列。写清楚。

### 5.2 新增 §III-D（修 R9-5、R9-6，同时修 **D**）

```latex
\subsection{Deployment}
\label{subsec:deployment}
At control time the model runs in closed loop: at \emph{every} control step the
current RGB observation $I_t$ and the task instruction are encoded, fused, and
passed once through the bottlenecked predictor. The predictor is never rolled out
over multiple steps and is never run once per episode; the residual is always
computed from the observation the robot is currently seeing, so it re-plans
implicitly as the scene changes.

The predictor emits a \emph{sequence of $N$ patch tokens}
$\hat z_{t+1} \in \mathbb{R}^{N \times D}$ in the frozen DINOv3 feature space; it
does not emit a class token. The residual is therefore formed patch-wise against
the current patch tokens, $\Delta_z = \hat z_{t+1} - z_t^{1:N}$, and retains spatial
structure. For 1-D continuous control we average $\Delta_z$ over the $N$ patches to
obtain a single $D$-dimensional vector, concatenate it with $z_t$ and the
proprioceptive state $q_t$, and feed the result to an MLP policy. For 2-D
spatial-action policies we reshape the $N$ tokens into a
$\sqrt{N} \times \sqrt{N} \times D$ map and fuse it with the policy's visual
backbone. Both settings use the same predictor output; only the read-out differs.

Retaining the predictor is cheap. It adds $X$ decoder blocks over $N{+}1$ tokens on
top of the encoder forward that every baseline already pays, i.e. $Y$\,ms per step
on a single $Z$ GPU, so the full perception stack runs at $W$\,Hz.
```

> `X/Y/Z/W` 需要你实测填。$X=4$（`depth_decoder=4`，dim 512）。
> **这段一定要有实测数字**——"retaining the predictor" 是你的核心卖点，
> 审稿人问部署频率，潜台词就是"每步都跑是不是太贵"。给个数就赢了。

⚠️ 注意：现在 §III-C 写的 "we flatten the \[CLS\] residual into a 1D vector"
**与代码不符**（预测器根本不输出 CLS）。上面这段已经改掉了。
如果 Franka Kitchen 那套确实用的是 CLS，请告诉我，写法要分开。

---

## 6. R9-7 / R9-8：视觉输入、相机视角、遮挡

### 6.1 三套实验各补一句输入规格（R9-7）

MT10 的我已经从代码里核实：

```latex
\noindent\textbf{Visual input.}
In Meta-World we render the \texttt{corner2} third-person camera at
$224\times224$, and the policy consumes a single RGB frame together with the
4-dimensional proprioceptive state (end-effector position and gripper aperture);
no wrist camera, no depth and no frame stacking are used.
```

Franka Kitchen 和真机的请照同样格式补：**相机是哪一个 / 装在哪 / 分辨率 / 单帧还是堆叠 /
有没有深度 / 本体感觉维度**。

### 6.2 真机相机（R9-7 + R9-8 的关键）

审稿人从 Fig.6 读出的是 "RealSense mounted near the end effector"。
从你的任务集（stack / pack / push piles，baseline 是 CLIPort、输出 affordance map、
工作空间 60×30 cm）判断，**几乎不可能是腕部相机，应该是固定俯视/第三人称**。
如果是这样，遮挡这条根本不成立，答复非常干净：

```latex
\noindent\textbf{Camera placement and occlusion.}
The RealSense is mounted on a fixed frame above the workspace and is not attached
to the end effector; it observes the full $60\times30$\,cm table throughout an
episode. The reviewer's concern about post-grasp occlusion does not arise in our
setup for two reasons. First, the camera is external, so the gripper occludes only
the region it is currently over rather than the whole workspace. Second, our
tabletop policy follows the CLIPort action abstraction: the pick pose and the
place pose are both inferred from the same pre-grasp observation, and the
place region is therefore resolved \emph{before} the object is picked up, so the
policy never needs to observe the placement region through a grasped object.
```

**如果实际就是腕部相机**，那这段不能这么写，必须改成说明"如何处理遮挡"，
并且要解释 affordance map 是怎么在腕部视角下生成的。请先确认。

### 6.3 图上必须动手（不只是改字）

- **Fig.6**：给相机画箭头标注 "fixed overhead RealSense D435"，标出安装支架，
  再加一个小 inset 放**相机真实拍到的画面**。这一张 inset 直接消灭 R9-7。
- **Fig.3**：caption 里的 "Obs" 明确写 "RGB observation from the same fixed overhead
  camera used by the policy"。

---

## 7. R9-9：场景初始化与评估次数

### 7.1 补一段 Evaluation protocol（真机部分，接在 §IV-E Setup 后）

```latex
\noindent\textbf{Evaluation protocol.}
At the start of every trial the target objects and the container are re-placed by
the experimenter at positions drawn uniformly over the reachable region of the
table, with yaw drawn uniformly from $[0, 2\pi)$, subject only to a minimum
separation of $\_$\,cm so that objects do not start in contact. $\_$ distractor
objects that are not mentioned in the instruction are placed in the same way. All
methods are evaluated on the \emph{same} set of initial configurations, replayed in
the same order, so that the comparison is paired rather than independent. A trial
is counted as a success if $\_$ within $\_$ seconds. We run $\_$ trials per task per
method and report success rate with $95\%$ Wilson confidence intervals.
```

### 7.2 5 次真的不够，这条改字救不了

- 5/5 和 4/5 的 95% Wilson 区间分别是 \[57,100\] 和 \[38,96\]——重叠得厉害，
  "100% vs 80%" 这种论断撑不住。审稿人说得对。
- 建议：**每任务至少 10 次**，7 任务 × 4 方法 = 280 trial。如果做不动，
  优先把 trial 数加在**你和最强 baseline（LaVA-Man）**上，弱 baseline 保持 5 次并注明。
- Fig.5（arm.pdf）的柱子上加 error bar，caption 写明 trial 数。
- MT10 那边 50 episodes/task + 独立 eval seed 已经写了，很好；
  但**单 seed** 这个 caveat 要在正文里明说一句，别让审稿人自己发现。

---

## 8. R9-10：R3M 的引用

审稿人自己说了 "I agree with the broader point"，所以**不要删观点，只改措辞**，
并且把他的话吸收进去——这是最容易让他满意的写法。

改 `intro.tex:80` 最后一句：

```latex
Second, the predictive component learned during pre-training is not carried into
policy learning. For predictive models this is literal: the decoder is discarded
once pre-training ends~\cite{radosavovic2023real, xiao2022masked}. For models whose
temporal signal comes from the objective rather than from a decoder---such as the
time-contrastive R3M~\cite{nair2022r3m}---nothing is thrown away, but the temporal
structure survives only implicitly, compressed into a static embedding. In both
cases the policy is handed features that describe the present and must re-identify,
from scratch, which visual changes are the ones that matter (Fig.~\ref{fig:teaser},
top).
```

Related Work §II-A 里 R3M 现在被归到 "extract robust features from static
observations via time-contrastive learning"，这句其实已经准确，不用改。

---

## 9. 图的修改

### 9.1 Fig.2 (`DiPredict_cropped.pdf`) —— 审稿人列了 4 条，逐条对应

| 审稿人要求 | 怎么改 |
|---|---|
| text backbone 没标 trainable/frozen | 全图上颜色图例：**蓝=trainable，灰+❄=frozen**。按代码：Teacher DINOv3 ❄、DistilBERT ❄、Student backbone ❄→🔥（标 "frozen for first $E_f$ epochs"）、`lang_proj`/`BiCrossAttn`/$F_\theta$ 🔥 |
| 缺箭头指示双向注意力的信息流 | BiCrossAttn 块内画**两条带方向的箭头**：$V\!\to\!L$ 在上、$L\!\to\!V$ 在下，并标序号 ①②，跟 Eq.(1)(2) 的顺序对上 |
| 图中 $F_{\text{intent}}$ 输入与 Eq.(4) 不一致 | 这是根子问题（上面的 **B**）。改完 Eq.(4) 后，图上把两条流画成**同一个 $F_\theta$ 方块被用两次**（或两个方块之间加 "shared $\theta$" 的连线），差别只画在 memory 那条边：full 流有一条 $V'\!\to\!F_\theta$ 的 memory 箭头，intent 流那条箭头**画成灰色打叉** |
| 建议加入标题中的术语 | 图上直接标 "intent-aligned"、"latent residual $\Delta_z$"、"asymmetric" |

再补两个图上没有但必须有的东西：
- **三个 loss 都要画出来**，$\mathcal{L}_{\text{cons}}$ 那条边在 full 侧画 **sg / ⊘** 符号。
- **部署时丢掉的那条流画成虚线灰色**，标 "training only"（按第 0 节 C，丢的是 full 流）。

### 9.2 Fig.3 (`aff_cropped.pdf`) affordance 可视化"奇怪"

审稿人没说哪里怪，通常是这几件事。建议一次全做：
- 用 **perceptually-uniform colormap**（viridis / inferno），别用 jet。
- 热力图 **overlay 在 RGB 上**（alpha≈0.5），而不是并排放两张图。
- **加 colorbar**，并在 caption 里写明归一化方式（per-map min-max？softmax？）。
- **pick map 和 place map 分行画并各自标注**，别混在一起。
- 每行标出**语言指令原文**。
- argmax 位置画一个十字/圆圈，让读者知道策略实际选了哪。
- 放**一个失败例子**。审稿人对"只挑好看的"很敏感，主动放一个反而加分。

### 9.3 附带风险：Fig.4 撑不起 Q1（审稿人没提，但下一轮很可能被抓）

`PAPER_HANDOFF.md` §5.2 里你自己提过：Fig.4 的 cross-attention 画在 **Kinetics 帧**上，
那是预训练域内，回答不了 Q1 的 "generalize to out-of-domain robotic embodiments"。
既然这一轮要改图，建议顺手把它**移出 Q1**，配文收窄成"域内语言-视觉 grounding"，
或者把 Q1 改成由 MT10/Franka 的数字来回答。**这条要你拍板。**

---

## 10. 落地顺序（按性价比）

1. **确认第 0 节的 A/B/C/D 四处到底以谁为准**（尤其 C：三套实验用的是不是同一条流）。
   这一步不做，下面全部白写。
2. §III-B 改写 + 伪代码框 + §III-C 首句改向 → 一次解决 AE-1、R9-5。
3. 新增 §III-D Deployment（含实测延迟数字）→ 解决 R9-5、R9-6、部分 R9-4。
4. §III-A 记号段 → R9-4。
5. Related Work 新增 latent action 小节 → R8-1。
6. §IV-A 的 $k$ 段 + 那条 $\cos(z_t,z_{t+k})$ 曲线 → R8-3。
7. 实验协议段（真机初始化、相机、遮挡、trial 数）→ R9-7/8/9。
8. Intro 的 R3M 措辞 → R9-10。
9. 消融段重写（MT10 矛盾）→ 预防下一轮。
10. 三张图重画。

## 11. 需要你决定 / 我做不了的

- [ ] **C：Franka 和真机用的是 intent 流还是 full 流？** 不一致的话是实质问题。
- [ ] **B：Eq.(4) 按代码改成"无 memory 瓶颈"，还是改代码去匹配"CLS 瓶颈"？**
- [ ] **真机相机到底是固定俯视还是腕部？** 决定 R9-8 怎么答。
- [ ] **真机 trial 数能不能从 5 提到 10+？** 这条改字救不了。
- [ ] **k 的 sensitivity 实验做不做？** 不做的话只能靠解释。
- [ ] **消融段是否按第 3 节重写？**（`PAPER_HANDOFF.md` 记着你不让改正文）
- [ ] **Fig.4 是否移出 Q1？**
- [ ] 提交前记得删掉 `intro.tex` 开头那段蓝色的 timeline `itemize`。
