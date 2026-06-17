# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

`datapostproc-rust` 是一个用于 DNS（Direct Numerical Simulation，直接数值模拟）数据后处理的 Rust 工具库。它通过 HDF5 格式读写仿真数据，支持超平板（hyperslab）分块读取，并提供流体力学后处理算法（如中心线定位）。

这是一个 Cargo workspace，包含两个 crate：
- **`datapostproc`**（`datapostproc/`）：主库，提供 HDF5 I/O、数据结构和数学算法。
- **`macro_struct`**（`macro_struct/`）：过程宏 crate，提供 `IterFields` derive 宏（开发中）。

## 常用命令

```bash
# 构建整个 workspace
cargo build

# 运行测试
cargo test

# 运行单个测试（按名称过滤）
cargo test <test_name>

# 检查代码（不生成二进制）
cargo check

# Clippy 静态分析
cargo clippy

# 格式化代码
cargo fmt
```

## 目录结构

```
datapostproc-rust/
├── Cargo.toml              # Workspace 配置，定义共享依赖
├── datapostproc/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # 导出 data、hdf5、math 模块
│       ├── main.rs         # 可执行入口（暂为空）
│       ├── data.rs         # 核心数据结构：H5File、Data、DNSInfo
│       ├── hdf5.rs         # HDF5 扩展：Block、BlockValue、DatasetHyperslabExt trait
│       ├── macros.rs       # 内部宏（当前为空）
│       ├── math/
│       │   ├── mod.rs
│       │   └── centerline.rs  # get_centerline()：通过 du/dz 零点定位中心线
│       └── utils/
│           └── io.rs       # 底层 HDF5 文件读取（read_hdf5_file，类型分发）
└── macro_struct/
    ├── Cargo.toml
    └── src/
        └── lib.rs          # IterFields 过程宏（开发中）
```

## 核心架构

### 数据分层

- **`utils/io.rs`**：最底层，直接操作 `hdf5-metno` crate，返回 `Hdf5Data` 枚举（涵盖 f32/f64/i32/i64 的标量和动态数组）。
- **`hdf5.rs`**：中间层，封装 HDF5 hyperslab 选择逻辑。`Block` 结构体描述分块读取参数（start/stride/count/block），`DatasetHyperslabExt` trait 为 `Dataset` 添加 `read_hyperslab` / `write_hyperslab` 方法。
- **`data.rs`**：高层，`H5File` 管理整个 HDF5 文件，`DNSInfo` 存储仿真元信息（网格尺寸 nx/ny/nz、物理尺寸 lx/ly/lz、雷诺数 re 等），`Data` 关联单个数据集与其分块配置。

### HDF5 分块（Hyperslab）约定

`BlockValue([start, stride, count, block])` 遵循 HDF5 hyperslab 语义：
- `start`：起始索引
- `stride`：两个 block 之间的步长（必须 > block 大小）
- `count`：block 数量
- `block`：每个 block 的大小

### 依赖说明

- `hdf5-metno`（alias `hdf5`）：HDF5 绑定，启用 blosc 压缩支持
- `ndarray`：N 维数组，启用 BLAS 后端
- `blosc-src`：编译期链接 blosc 压缩库

## 代码风格

- 使用 Rust 2024 edition
- 错误处理统一使用 `hdf5::Result` / `hdf5::Error`，通过 `.into()` 构造字符串错误
- 带状态的结构体（如 `Block`）使用 `validated` 标志位防止未初始化使用
- 数学算法放在 `math/` 子模块，每个算法独立文件
