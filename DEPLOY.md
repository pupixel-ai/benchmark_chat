# 记忆工程项目部署指南

本文优先说明如何把当前仓库部署到 AWS EC2，默认采用：

- 一台 Ubuntu EC2
- `systemd` 守护后端和前端
- `nginx` 做同域反向代理
- 单机 SQLite 起步

这样改动最少，也最适合先把服务跑通。后续如果你要升级成 `EC2 + RDS + S3`，本仓库也已经支持通过环境变量切换。

如果你要直接走新版本的 AWS 生产拓扑，请看：

- `deploy/aws/README.md`
- `deploy/aws/backend.Dockerfile`
- `deploy/aws/frontend.Dockerfile`
- `deploy/aws/control-plane.env.example`
- `deploy/aws/worker.env.example`

如果你要继续做高安全的 Phase 2 worker 架构，请看：

- `deploy/ec2/PHASE2_WORKER_TEMPLATE.md`
- `deploy/ec2/worker-user-data.sh.example`
- `deploy/ec2/memory-engineering-worker.service`
- `deploy/ec2/worker.env.example`

## 部署架构

```text
浏览器
  ↓ https://memory.example.com
Nginx (80/443)
  ├─ /        → Next.js 前端 (127.0.0.1:3000)
  └─ /api/*   → FastAPI 后端 (127.0.0.1:8000)

后端本地依赖
  ├─ SQLite: runtime/local_preview.db
  ├─ 任务目录: runtime/tasks/
  └─ 模型/缓存: cache/、runtime/
```

## 上线前准备

### 1. 创建 EC2

推荐先用：

- Ubuntu 22.04 LTS 或 24.04 LTS
- 至少 2 vCPU / 8 GB 内存
- 系统盘至少 40 GB

说明：

- 这个项目会做图片解码、人脸识别和本地文件落盘，内存太小会明显拖慢。
- 初次运行 InsightFace / ONNX 相关模型时，还会有额外下载和缓存。

### 2. 安全组开放端口

至少开放：

- `22`：你的公网 IP
- `80`：`0.0.0.0/0`
- `443`：`0.0.0.0/0`

如果你暂时不接域名，也可以临时开放 `3000` 和 `8000` 做排查，但正式环境不建议长期暴露。

### 3. 域名

建议先准备一个域名，比如：

- `memory.example.com`

然后把它解析到 EC2 公网 IP。

## Step By Step

以下命令默认在一台全新 Ubuntu EC2 上执行，系统用户为 `ubuntu`。

### Step 1. 连接服务器

```bash
ssh -i /path/to/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git nginx certbot python3-certbot-nginx curl
```

检查版本：

```bash
python3 --version
node --version
npm --version
```

然后安装 Node.js 20：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 20
nvm alias default 20
node --version
npm --version
```

当前仓库里的 Next.js 版本建议使用 Node 20.9 或更高版本。

### Step 3. 拉取代码

```bash
cd /opt
sudo git clone <你的仓库地址> memory_engineering
sudo chown -R ubuntu:ubuntu /opt/memory_engineering
cd /opt/memory_engineering
```

如果你不是通过 Git 部署，也可以先在本地上传压缩包，再在服务器解压到 `/opt/memory_engineering`。

### Step 4. 创建 Python 虚拟环境并安装后端依赖

```bash
cd /opt/memory_engineering
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5. 安装前端依赖并构建

```bash
cd /opt/memory_engineering/frontend
npm install
cp /opt/memory_engineering/deploy/ec2/frontend.env.example .env.production
npm run build
```

说明：

- `frontend/.env.production` 里默认让前端走同域 `/api`，这样 Nginx 反代后不需要单独处理跨域。

### Step 6. 配置后端环境变量

```bash
cd /opt/memory_engineering
cp deploy/ec2/backend.env.example .env
```

然后编辑 `/opt/memory_engineering/.env`，至少填这些：

```bash
USE_API_PROXY=false
GEMINI_API_KEY=你的真实密钥
AMAP_API_KEY=你的真实密钥
FRONTEND_ORIGIN=https://memory.example.com
COOKIE_SECURE=true
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_RELOAD=false
DATABASE_URL=sqlite:////opt/memory_engineering/runtime/local_preview.db
```

如果你使用代理模式，则改成：

```bash
USE_API_PROXY=true
API_PROXY_URL=https://your-proxy-host.example.com
API_PROXY_KEY=your_proxy_api_key_here
API_PROXY_MODEL=gemini-2.0-flash
```

### Step 7. 先在命令行手动验证后端

```bash
cd /opt/memory_engineering
source .venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

另开一个 SSH 窗口检查：

```bash
curl http://127.0.0.1:8000/api/health
```

如果返回 JSON，说明后端基础链路正常。

### Step 8. 手动验证前端

```bash
cd /opt/memory_engineering/frontend
npm run start
```

另开一个 SSH 窗口检查：

```bash
curl -I http://127.0.0.1:3000
```

如果返回 `200` 或 `307` 一类状态码，说明前端可启动。

### Step 9. 配置 systemd 后端服务

```bash
sudo cp /opt/memory_engineering/deploy/ec2/memory-engineering-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable memory-engineering-backend
sudo systemctl start memory-engineering-backend
sudo systemctl status memory-engineering-backend
```

看日志：

```bash
journalctl -u memory-engineering-backend -n 100 --no-pager
```

### Step 10. 配置 systemd 前端服务

```bash
sudo cp /opt/memory_engineering/deploy/ec2/memory-engineering-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable memory-engineering-frontend
sudo systemctl start memory-engineering-frontend
sudo systemctl status memory-engineering-frontend
```

看日志：

```bash
journalctl -u memory-engineering-frontend -n 100 --no-pager
```

### Step 11. 配置 Nginx

先复制模板：

```bash
sudo cp /opt/memory_engineering/deploy/ec2/nginx-memory-engineering.conf /etc/nginx/sites-available/memory-engineering
```

编辑 `/etc/nginx/sites-available/memory-engineering`，把：

```nginx
server_name memory.example.com;
```

替换成你的真实域名。

启用站点：

```bash
sudo ln -sf /etc/nginx/sites-available/memory-engineering /etc/nginx/sites-enabled/memory-engineering
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Step 12. 申请 HTTPS 证书

确认域名已经解析到 EC2 后执行：

```bash
sudo certbot --nginx -d memory.example.com
```

成功后检查：

```bash
curl -I https://memory.example.com
curl https://memory.example.com/api/health
```

### Step 13. 首次登录和上传测试

浏览器打开：

```text
https://memory.example.com
```

然后按顺序测试：

1. 注册一个测试账号
2. 上传 1 到 3 张照片
3. 观察任务是否能进入 `completed`
4. 检查人脸框、预览图、任务详情是否能正常加载

## 生产注意事项

### 1. 后端只建议单进程运行

当前任务执行依赖：

- FastAPI `BackgroundTasks`
- 本地任务目录 `runtime/tasks/`
- 本地 SQLite

所以在单机 EC2 方案里，后端建议保持：

- 1 个 `uvicorn` 进程
- 不要直接开多 worker

否则你会很难排查任务目录、锁竞争和本地状态一致性问题。

### 2. 首次模型下载会比较慢

第一次触发人脸识别时，相关模型可能会在服务器上下载或初始化，属于正常现象。

### 3. EC2 重启后的持久化

以下目录建议保留在实例磁盘上：

- `/opt/memory_engineering/runtime`
- `/opt/memory_engineering/cache`

如果你后续改成 Auto Scaling 或多机部署，就不要依赖本地磁盘了，建议迁移到：

- 数据库：RDS MySQL
- 文件：S3

### 4. 对象存储是可选项

当前仓库在没有对象存储配置时，也可以直接从后端读取本地任务文件。

如果你想切换到 S3 或兼容存储，在 `.env` 填这些变量即可：

```bash
OBJECT_STORAGE_BUCKET=your-bucket
OBJECT_STORAGE_ENDPOINT=https://s3.ap-southeast-1.amazonaws.com
OBJECT_STORAGE_REGION=ap-southeast-1
OBJECT_STORAGE_ACCESS_KEY_ID=your_access_key
OBJECT_STORAGE_SECRET_ACCESS_KEY=your_secret_key
OBJECT_STORAGE_PREFIX=tasks
OBJECT_STORAGE_ADDRESSING_STYLE=auto
```

### 5. 数据库升级到 MySQL / RDS

把 `.env` 里的：

```bash
DATABASE_URL=sqlite:////opt/memory_engineering/runtime/local_preview.db
```

改成：

```bash
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/DATABASE
```

即可切换到 MySQL。

## 常用排查命令

### 看后端日志

```bash
journalctl -u memory-engineering-backend -f
```

### 看前端日志

```bash
journalctl -u memory-engineering-frontend -f
```

### 看 Nginx 日志

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 看服务状态

```bash
systemctl status memory-engineering-backend
systemctl status memory-engineering-frontend
systemctl status nginx
```

### 重启服务

```bash
sudo systemctl restart memory-engineering-backend
sudo systemctl restart memory-engineering-frontend
sudo systemctl reload nginx
```

## 更新发布流程

每次代码更新后，按这个顺序执行：

```bash
cd /opt/memory_engineering
git pull
source .venv/bin/activate
pip install -r requirements.txt
cd frontend
npm install
npm run build
sudo systemctl restart memory-engineering-backend
sudo systemctl restart memory-engineering-frontend
```

## Railway 说明

如果你之后仍然想部署到 Railway，仓库原有能力仍可用：

- 后端优先读取 `DATABASE_URL`
- 未设置时会尝试 Railway 注入的 MySQL 变量
- 对象存储优先读取 `BUCKET / ENDPOINT / ACCESS_KEY_ID / SECRET_ACCESS_KEY`

但对当前这个仓库来说，EC2 更适合做长任务和大图处理。
