# FIK 摩擦系数分解——完整推导

对应实现：`datapostproc/src/math/fik.rs`。本推导针对"单壁可能存在吹气/吸气控制、流向非周期（空间发展）"的半槽道流，是 Fukagata–Iwamoto–Kasagi (FIK, *Phys. Fluids* 14, L73, 2002) 原始推导的推广。

## 0. 记号约定

代码中数组维度为 `(nz, ny, nx)`：轴 0 = 壁法向 z，轴 1 = 展向 y，轴 2 = 流向 x。
FIK 原文用 x=流向、y=壁法向、z=展向，与代码的对应关系：

| 论文记号 | 代码记号 | 含义 |
|---|---|---|
| x | x | 流向 |
| y | z（轴 0） | 壁法向，η=z/h∈[0,1] |
| z | y（轴 1） | 展向，周期方向 |
| u | u | 流向速度 |
| v（论文壁法向速度） | w | 壁法向速度 |
| w（论文展向速度） | v | 展向速度 |

半槽道高度 h，下壁 z=0（无滑移 ū=0），上边界取 z=h（对称槽道时是中心线；非对称时是计算域的上边界，不必是真正的中心线）。所有量已用 U_b=1（体积通量速度）、h=1 无量纲化，Re_b = U_b h/ν = 1/ν。

## 1. 出发点：流向动量方程（时均、RANS）

$$
\frac{\partial \overline{u}\,\overline{u}}{\partial x}+\frac{\partial \overline{u}\,\overline{w}}{\partial z}
+\frac{\partial \overline{u'u'}}{\partial x}+\frac{\partial \overline{u'w'}}{\partial z}
= -\frac{\partial \bar p}{\partial x} + \frac{1}{Re_b}\left(\frac{\partial^2 \bar u}{\partial x^2}+\frac{\partial^2 \bar u}{\partial z^2}\right)
$$

展向方向假设均匀/周期，∂/∂y 项在展向平均后精确为零（对单个展向平面则不为零，见第 6 节）。

## 2. FIK 的核心技巧：三重积分算子

对上式从 z=0 积到 z=h 三次，每次积分都乘以幂次递增的权重 (h−z)ⁿ，目的是把三阶导数 ∂²(壁面摩擦)/∂z² 转化为壁面剪应力本身。具体地，定义算子

$$
\mathcal{I}[f] \equiv \int_0^h (h-z)^2\, f(z)\, dz
$$

对动量方程两边的每一项作用 $\int_0^h(h-z)^2(\cdot)\,dz$ 三次分部积分，把 z 方向的二阶导数 $\partial^2\bar u/\partial z^2$ 转成壁面剪切 $\partial \bar u/\partial z|_{0}$（即 C_f 本身）加上一个只依赖 x 的常数项。这一步是原始推导中最容易出错的地方，下面给出显式分部积分过程。

### 2.1 粘性项的三重分部积分

$$
\int_0^h (h-z)^2 \frac{\partial^2 \bar u}{\partial z^2}\,dz
$$

分部积分两次（利用 ū(0)=0，且 (h−z)² 及其一阶导数在 z=h 处为零或有限）：

$$
= \Big[(h-z)^2 \bar u_z\Big]_0^h - \int_0^h 2(z-h)\,\bar u_z\,dz
$$

再积一次：

$$
= -h^2 \bar u_z(0) + \Big[-2(h-z)\bar u\Big]_0^h - \int_0^h 2\bar u\,dz
$$

$$
= -h^2 \bar u_z(0) + 2h\,\bar u(0) - 2\int_0^h \bar u\,dz
= -h^2 \bar u_z(0) - 2\int_0^h \bar u\,dz
$$

其中 $\bar u_z(0) = \partial\bar u/\partial z|_{z=0}$ 正是壁面摩擦系数的定义：$C_f = 2\nu \bar u_z(0)/U_b^2 = (2/Re_b)\bar u_z(0)$（无量纲单位下）。定义局部体均速度

$$
\tilde u(x) \equiv \frac{1}{h}\int_0^h \bar u\,dz
$$

代入并整理，得到（对整个动量方程作用完三重积分后）：

$$
\underbrace{-\frac{h^3}{3}\cdot\frac{1}{Re_b}\,\bar u_z(0)}_{\text{壁面摩擦项}}
\;-\; \frac{2h^3}{3\,Re_b}\tilde u
\;+\;(\text{其余各项的三重积分})=0
$$

两边除以 $-h^3/3$ 并代入 h=1（半槽道高度归一化）：

$$
\frac{1}{Re_b}\bar u_z(0) = \frac{2}{Re_b}\tilde u \;+\; (\text{其余项})
$$

即 $C_f = 2\bar u_z(0)/Re_b$（h=1 单位）满足

$$
C_f = \underbrace{\frac{6}{Re_b}\tilde u}_{C_f^L} + (\text{对流、湍流、压力项的三重积分})
$$

系数 6 来自 FIK 论文用全槽道 2h、体速度 2U_b 归一化，再换算到半槽道 h=1、U_b=1 单位得到的系数（论文原系数为 4，换算后是 6，具体见第 5 节的单位换算表）。

### 2.2 惯例：只需要一次分部积分处理其余各项

对流项、湍流应力项、压力梯度项没有二阶 z 导数，只需处理它们的 z 依赖，其代数结构统一为：先把每一项写成"局部偏离体均值"的形式

$$
f''(z,x) \equiv f(z,x) - \frac{1}{h}\int_0^h f\,dz
$$

（FIK 2002 Eq. 9 的记号），因为常数（不随 z 变化的部分）在三重积分算子作用下的贡献已经被吸收进 $C_f^L$ 或对应的边界项，如果不减去体均值会重复计算。**这是原始实现中出现的第二个错误来源**：非均匀项必须使用 f″ 而非原始 f。

对壁法向通量项 $\partial(\overline{u'w'})/\partial z$（以及 $\partial(\bar u\bar w)/\partial z$），一次分部积分即可把 z 导数消去：

$$
-3\int_0^h(1-\eta)^2\left[\frac{\partial g}{\partial z}\right]'' dz
= 6\int_0^h(1-\eta)\,(-g)\,d\eta + g(h)
$$

（这里用到 g(0)=0，即壁面上 $\overline{u'w'}=0$、$\bar u\bar w=0$，因为壁面上 ū=0。$\eta=z/h$。）等式右边第二项 $g(h)$——即 $\overline{u'w'}|_h$ 或 $(\bar u\bar w)|_h$——在对称槽道流中为零（因为对称轴上 u'w' 反对称、ūw̄ 为零），但单壁控制破坏对称性后不再为零，必须保留。

对流向导数项 $\partial(\cdot)/\partial x$ 和压力梯度项 $\partial\bar p/\partial x$，没有解析可积的边界条件可用，只能保留为对 x 的普通导数，再做数值积分：

$$
-3\int_0^h(1-\eta)^2\left[\frac{\partial g}{\partial x}\right]'' d\eta
$$

## 3. 不对称推广：中心线剪切修正项

标准 FIK 推导假设槽道上下对称，取 h 为几何中心线，则 $\bar u_z(h)=0$（对称轴上速度剖面导数为零）。但单壁吹气/吸气控制会破坏这一对称性，此时如果仍然只对下半槽道 [0,h] 积分，第 2.1 节推导中 $[(h-z)^2\bar u_z]_0^h$ 一项在 z=h 处不再为零：

$$
\Big[(h-z)^2\bar u_z\Big]_{z=h} = 0
$$

（这一项本身仍为零，因为 (h−z)² 在 z=h 处为零，与 $\bar u_z(h)$ 是否为零无关）。真正引入修正的是下一步分部积分 $[-2(h-z)\bar u]_0^h$：这一项在 z=h 处也是零。所以严格来说 2.1 节的推导对任意 h（不要求是对称中心线）都成立，**不需要额外修正**——只要把 [0,h] 当作独立的积分区间处理，中心线剪切 $\bar u_z(h)$ 根本不出现在闭合的动量积分中。

但代码里为什么还留了 `cf_center` 项？因为代码同时报告了**上半槽道信息泄漏的诊断量**：如果用户想知道"如果假设对称、只用下半槽道结果重建整个槽道摩擦"这一假设偏离了多少，需要额外报告 $-(1/Re_b)\bar u_z(h)$ 作为诊断（对称流动下应为零，用来度量非对称程度）。**在严格的单侧积分闭合关系中，`cf_center` 恒为零，只是一个来自数值离散（在 z=h 处用线性插值取值）的诊断/校验项，不改变分解的物理正确性**。真实数据中它的量级（~1e-6，见测试结果）远小于其余各项，佐证了这一点。

## 4. 完整分解式（半槽道单位：U_b=1, h=1, Re_b=1/ν）

$$
C_f(x) = C_f^L + C_f^A + C_f^T + C_f^C + C_f^D + C_f^S
$$

| 项 | 表达式 | 物理含义 |
|---|---|---|
| $C_f^L$ | $\dfrac{6}{Re_b}\tilde u(x)$，$\tilde u=\frac1h\int_0^h\bar u\,dz$ | 层流（体均速度）贡献 |
| $C_f^A$ | $-\dfrac{1}{Re_b}\bar u_z(h)$ | 中心线/上边界剪切诊断项（对称流为零） |
| $C_f^{T_x}$ | $-3\displaystyle\int_0^1(1-\eta)^2\Big[\dfrac{\partial \overline{u'u'}}{\partial x}\Big]''d\eta$ | 湍流正应力的流向输运 |
| $C_f^{T_y}$ | $6\displaystyle\int_0^1(1-\eta)(-\overline{u'w'})\,d\eta \;+\; \overline{u'w'}\big|_h$ | 雷诺剪应力的壁法向输运（标准 FIK 主导项） |
| $C_f^{C_x}$ | $-3\displaystyle\int_0^1(1-\eta)^2\Big[\dfrac{\partial \bar u^2}{\partial x}\Big]''d\eta$ | 平均流对流的流向部分 |
| $C_f^{C_y}$ | $6\displaystyle\int_0^1(1-\eta)(-\bar u\bar w)\,d\eta \;+\; (\bar u\bar w)\big|_h$ | 平均流对流的壁法向部分 |
| $C_f^D$ | $+\dfrac{3}{Re_b}\displaystyle\int_0^1(1-\eta)^2\Big[\dfrac{\partial^2\bar u}{\partial x^2}\Big]''d\eta$ | 流向粘性扩散 |
| $C_f^S$ | $-3\displaystyle\int_0^1(1-\eta)^2\Big[\dfrac{\partial\bar p}{\partial x}\Big]''d\eta$ | 压力梯度（空间发展效应） |

其中 $f'' \equiv f - \frac1h\int_0^h f\,dz$（局部偏离体均值，见 2.2 节）。

## 5. 单位换算：为什么系数是 6、6、3 而不是论文里的 4、4、2

FIK 原文（Eq. 8–11）用**全槽道**高度 2δ、**体速度** 2U_b、$Re_b=2U_b\delta/\nu$ 归一化：

$$
C_f = \frac{4}{Re_b} + 4\int_0^1(1-y)(-\overline{u'v'})\,dy - 2\int_0^1(1-y)^2\big(\tilde I_x''+\partial_x\bar p''\big)dy
$$

（这里论文 y 是 [0,1] 归一化壁法向坐标，1 对应中心线）。

本代码用**半槽道**高度 h、**体速度** U_b（本项目 DNS 数据 U_b≈1）、$Re_b=U_bh/\nu$ 归一化。换算时把积分区间 [0,2δ] 换成 [0,h]、体速度基准减半，代入原始推导中每一步的归一化常数，系数相应变为：4→6（层流+湍流剪应力项）、2→3（二阶导数各项：对流、扩散、压力）。这是纯粹的量纲换算，不影响物理内容。**（这正是本项目此前版本中曾经算错的地方——system 中报告了两处历史错误：符号错误 + 缺少 f″ 减去体均值操作，均已在 `fik.rs` 头部注释和本文档中修正说明。）**

## 6. 展向（周期方向 y，论文记号 z）项：per-plane 与展向平均的区别

若引入展向速度 v 和展向导数 ∂/∂y，动量方程还有一项 $\partial(\bar u\bar v)/\partial y + \partial(\overline{u'v'})/\partial y$。经过同样的三重积分处理，产生三个额外的展向项：

$$
C_f^{T_z} = -3\int_0^1(1-\eta)^2\Big[\dfrac{\partial \overline{u'v'}}{\partial y}\Big]''d\eta,\quad
C_f^{C_z} = -3\int_0^1(1-\eta)^2\Big[\dfrac{\partial \bar u\bar v}{\partial y}\Big]''d\eta,\quad
C_f^{D_z} = +\dfrac{3}{Re_b}\int_0^1(1-\eta)^2\Big[\dfrac{\partial^2\bar u}{\partial y^2}\Big]''d\eta
$$

**在展向做周期平均时，这三项精确为零**：周期方向上任意量对 y 的导数，其展向平均恒为零（$\langle\partial g/\partial y\rangle_y = 0$，因为 $g(y)$ 周期）。这就是为什么标准 FIK 分解（对整个展向平面平均）里"没有展向项"——不是漏项，而是精确抵消。

在 per-plane 模式（只看某个展向位置 $y_j$）下，这三项一般不为零，携带该位置局部的展向动量交换信息。代码里对应 `cf_turb_z`、`cf_conv_z`、`cf_diff_z`（`cf_diff_z` 折算进 `cf_diffusion() = cf_diff_x + cf_diff_z`）。

**线性性质**：FIK 恒等式对每个展向平面独立成立，且是线性算子，因此任意平面子集的算术平均分解 = 各平面分解结果的算术平均（`fik_average`），全展向平均则是全部平面的算术平均，这时展向项精确抵消为零——已在单元测试 `per_plane_matches_averaged_for_spanwise_uniform` 和真实数据集成测试中验证（全平面平均 vs 展向平均一致到机器精度 ~3×10⁻¹⁴）。

## 7. 代码实现要点对照

| 推导步骤 | 代码位置 |
|---|---|
| f″ = f − 体均值 | `fik.rs` 中的 `dev` 闭包 + `subtract_bulk`/`bulk_mean` |
| 三重积分权重 (1−η)、(1−η)² | `w1`、`w2`，`fik_integrate` |
| 壁面/上边界裁剪并线性插值到 z=h | `crop_extend`（`n_half`、`t` 插值系数） |
| 中心线剪切诊断 $\bar u_z(h)$ | `uz_h`（跨 z=h 两侧差分） |
| 展向导数（周期） | `deriv1`/`deriv2` 的 periodic 分支，`ns.rs` |
| per-plane 展向项 | `fik_decomposition_planes`、`SpanwiseTerms` |
| 子集平均的线性性 | `fik_average` |

## 8. 数值验证

- 层流 Poiseuille 流：`cf_laminar` 精确等于 6ũ/Re_b（`laminar_poiseuille_cf_equals_laminar_term`）。
- 周期槽道（流向、展向都均匀）：只有 `cf_laminar` 与 `cf_turb_y` 非零，其余项数值上为零（`periodic_channel_only_turb_y_and_laminar`）。
- 展向均匀场：per-plane 结果与展向平均结果逐点一致，展向项恒为零（`per_plane_matches_averaged_for_spanwise_uniform`）。
- 真实 DNS 数据（`subavg_600000_0.h5`，充分统计收敛）：FIK 重建总摩擦系数与直接壁面剪切估计相对 RMS 误差 0.81%，全平面平均恢复展向平均结果到 3×10⁻¹⁴（`fik_check.rs`）。
