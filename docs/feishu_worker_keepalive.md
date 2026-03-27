# Feishu Worker 本机长期保活

## 目标

让飞书长链接 worker 在本机通过 `launchd` 常驻：

- 登录后自动拉起
- 异常退出自动重启
- 可用命令一键安装/卸载

## 安装

```bash
cd /Users/vigar07/Desktop/benchmark_chat
chmod +x scripts/launchd/*.sh
./scripts/launchd/install_feishu_long_connection_launchd.sh
```

说明：

- 如果工程目录在 `Desktop/Documents/Downloads`（macOS 受保护目录），安装脚本会自动同步一份运行时镜像到
  `/Users/vigar07/.codex_local/runtime/benchmark_chat`
- `launchd` 实际会从该镜像目录启动，避免后台进程读取受保护目录时报 `Operation not permitted`

## 检查状态

```bash
launchctl print gui/$(id -u)/com.benchmark_chat.feishu_long_connection | sed -n '1,120p'
```

日志：

- `/Users/vigar07/.codex_local/runtime/benchmark_chat/.codex_local/logs/launchd.feishu_long_connection.out.log`
- `/Users/vigar07/.codex_local/runtime/benchmark_chat/.codex_local/logs/launchd.feishu_long_connection.err.log`

查看实际生效根目录：

```bash
launchctl print gui/$(id -u)/com.benchmark_chat.feishu_long_connection | rg "FEISHU_WORKER_ROOT|working directory"
```

## 重启 worker

```bash
launchctl kickstart -k gui/$(id -u)/com.benchmark_chat.feishu_long_connection
```

## 卸载

```bash
cd /Users/vigar07/Desktop/benchmark_chat
./scripts/launchd/uninstall_feishu_long_connection_launchd.sh
```

## 说明

runner 进程通过 `caffeinate -dims` 防止锁屏和空闲导致休眠。  
在 MacBook 合盖场景下，系统仍可能进入睡眠（尤其不满足外接电源/外接显示条件时）。若要求严格 24x7 在线，建议部署到常在线主机。
