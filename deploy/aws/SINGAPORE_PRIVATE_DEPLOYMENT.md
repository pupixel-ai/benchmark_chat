# Singapore 私网从零部署手册

本手册面向当前 `v0317` 方案，目标是：

- 区域：`ap-southeast-1`
- 网络：AWS 私网 + 已打通的 IPSec
- 架构：
  - `app EC2`: frontend + backend + face service
  - `memory EC2`: Neo4j + Milvus
  - `RDS MySQL`: 应用数据库 + artifact catalog
  - `S3`: task artifacts

## 0. 最终拓扑

```text
On-prem / Office
   ↓ IPSec
AWS VPC (ap-southeast-1)
   ├─ private app EC2
   │    ├─ Next.js frontend
   │    ├─ FastAPI backend
   │    └─ face service / pipeline
   ├─ private memory EC2
   │    ├─ Neo4j
   │    └─ Milvus
   ├─ private RDS MySQL
   └─ S3 bucket
```

## 1. AWS Console 创建顺序

### 1.1 确认网络前提

你已经有 IPSec 到 AWS 内网，因此先确认：

- VPC 已存在
- 目标子网网段为 `10.60.1.0/24` 或你当前实际使用的私网段
- Route table 已经让 on-prem 网段和 AWS 私网互通

参考：
- [AWS Site-to-Site VPN 路由配置](https://docs.aws.amazon.com/vpn/latest/s2svpn/SetUpVPNConnections.html)
- [VPC 子网路由表](https://docs.aws.amazon.com/vpc/latest/userguide/subnet-route-tables.html)

### 1.2 创建 S3 Bucket

1. 打开 S3 Console
2. Create bucket
3. Region 选 `Asia Pacific (Singapore) ap-southeast-1`
4. Bucket name 例如：
   - `memory-engineering-artifacts-prod`
5. Block Public Access：保持开启
6. 创建完成

### 1.3 创建 RDS MySQL

1. 打开 RDS Console
2. Create database
3. Engine type 选 `MySQL`
4. Version 选 `MySQL 8.0`
5. Template 选 `Production`
6. DB instance identifier：
   - `memory-engineering-db`
7. Master username：
   - `memory_app`
8. 记录数据库密码
9. Instance class：
   - 起步 `db.t4g.medium`
10. Storage：
   - `gp3`, `100 GiB` 起步
11. Connectivity：
   - VPC 选目标 VPC
   - Public access 选 `No`
   - Security group 单独创建 `memory-rds-sg`
12. 创建完成后记录：
   - `RDS endpoint`

参考：
- [RDS MySQL 入门](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_MySQL.html)
- [RDS 安全组访问控制](https://docs.aws.amazon.com/AmazonRDS/latest/gettingstartedguide/security-groups.html)

### 1.4 创建 memory EC2

1. 打开 EC2 Console
2. Launch instance
3. Name：
   - `memory-v0317`
4. AMI：
   - `Ubuntu Server 24.04 LTS`
5. Instance type：
   - 推荐 `r7i.xlarge`
   - 最低可先 `m7i.xlarge`
6. Key pair：
   - 选你的 `.pem` 对应 key
7. Network settings：
   - VPC：目标 VPC
   - Subnet：目标私有子网
   - Auto-assign public IP：`Disable`
8. Security group：
   - 新建 `memory-v0317-sg`
9. Inbound rules：
   - `SSH 22` from 你的办公/机房网段 CIDR
   - `Custom TCP 7474` from 办公网段 CIDR
   - `Custom TCP 9091` from 办公网段 CIDR
   - `Custom TCP 7687` from `app-v0317-sg`
   - `Custom TCP 19530` from `app-v0317-sg`
10. Storage：
   - `gp3 300 GiB`
11. Launch
12. 记录这台机器的 `Private IPv4`

### 1.5 创建 app EC2

1. Launch instance
2. Name：
   - `app-v0317`
3. AMI：
   - `Ubuntu Server 24.04 LTS`
4. Instance type：
   - 推荐 `m7i.xlarge`
   - 至少 `m7i.large`
5. Key pair：
   - 同上
6. Network settings：
   - VPC：目标 VPC
   - Subnet：目标私有子网
   - Auto-assign public IP：`Disable`
7. Security group：
   - 新建 `app-v0317-sg`
8. Inbound rules：
   - `SSH 22` from 你的办公/机房网段 CIDR
   - `Custom TCP 3000` from 办公网段 CIDR
   - `Custom TCP 8000` from 办公网段 CIDR
9. Storage：
   - `gp3 200 GiB`
10. Launch
11. 记录这台机器的 `Private IPv4`

### 1.6 安全组互通

#### `memory-rds-sg`
- 入站：
  - `3306` from `app-v0317-sg`

#### `memory-v0317-sg`
- 入站：
  - `22` from 办公网段 CIDR
  - `7474` from 办公网段 CIDR
  - `9091` from 办公网段 CIDR
  - `7687` from `app-v0317-sg`
  - `19530` from `app-v0317-sg`

#### `app-v0317-sg`
- 入站：
  - `22` from 办公网段 CIDR
  - `3000` from 办公网段 CIDR
  - `8000` from 办公网段 CIDR

## 2. IAM 配置

### 2.1 app EC2 instance profile

创建一个 EC2 role，例如：
- `memory-app-ec2-role`

至少给它这些 S3 权限：
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`
- `s3:ListBucket`

参考：
- [EC2 使用 IAM role 访问 AWS 资源](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2.html)
- [S3 访问控制](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-management.html)

把这个 role 挂到 `app-v0317` EC2。

### 2.2 memory EC2 instance profile

Neo4j / Milvus 本身不强依赖 AWS API，通常不需要额外权限。  
如果你想让它也读 S3 备份，再单独扩权限。

## 3. memory EC2 从 0 到可用的命令

假设 memory EC2 私网 IP 是 `10.60.1.91`：

```bash
ssh -i /path/to/your.pem ubuntu@10.60.1.91

sudo apt update
sudo apt install -y git docker.io docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker

cd /opt
sudo git clone https://github.com/pupixel-ai/benchmark_chat.git memory_engineering
sudo chown -R ubuntu:ubuntu /opt/memory_engineering

cd /opt/memory_engineering
git checkout main
git pull --ff-only origin main
```

### 3.1 启动 Neo4j

```bash
cd /opt/memory_engineering/deploy/aws/neo4j
cp neo4j.env.example neo4j.env
nano neo4j.env
```

编辑：

```env
NEO4J_AUTH=neo4j/YOUR_STRONG_PASSWORD
```

启动：

```bash
docker compose up -d
docker compose ps
docker logs memory-neo4j --tail 100
```

### 3.2 启动 Milvus

```bash
cd /opt/memory_engineering/deploy/aws/milvus
cp milvus.env.example milvus.env
nano milvus.env
```

编辑：

```env
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=YOUR_STRONG_SECRET
```

启动：

```bash
docker compose up -d
docker compose ps
docker logs memory-milvus --tail 100
```

### 3.3 memory EC2 探活

```bash
docker ps
ss -ltnp | egrep '7474|7687|19530|9091'
curl http://127.0.0.1:9091/healthz
```

官方参考：
- [Neo4j Docker 配置](https://neo4j.com/docs/operations-manual/current/docker/configuration/)
- [Neo4j Docker 运维与内存建议](https://neo4j.com/docs/operations-manual/current/docker/operations/)
- [Neo4j 内存配置](https://neo4j.com/docs/operations-manual/current/performance/memory-configuration/)
- [Milvus Docker Compose 配置](https://milvus.io/docs/configure-docker.md)

## 4. app EC2 从 0 到可用的命令

假设 app EC2 私网 IP 是 `10.60.1.90`，memory EC2 私网 IP 是 `10.60.1.91`：

```bash
ssh -i /path/to/your.pem ubuntu@10.60.1.90

sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git curl
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 20
nvm alias default 20

cd /opt
sudo git clone https://github.com/pupixel-ai/benchmark_chat.git memory_engineering
sudo chown -R ubuntu:ubuntu /opt/memory_engineering

cd /opt/memory_engineering
git checkout main
git pull --ff-only origin main

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd /opt/memory_engineering/frontend
npm install

cd /opt/memory_engineering
cp .env.example .env
nano .env
```

## 5. app EC2 `.env` 模板

以下是建议的完整模板，直接替换真实值：

```env
APP_VERSION=v0317
DEFAULT_TASK_VERSION=v0317
MAX_UPLOAD_PHOTOS=5000

MODEL_PROVIDER=bedrock
VLM_PROVIDER=openrouter
LLM_PROVIDER=openrouter
BEDROCK_REGION=ap-southeast-1
BEDROCK_VLM_MODEL_PRIMARY=amazon.nova-pro-v1:0
BEDROCK_VLM_MODEL_FALLBACK=amazon.nova-pro-v1:0
BEDROCK_VLM_MODEL_POLICY=primary
BEDROCK_LLM_MODEL=anthropic.claude-sonnet-4-6
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://10.60.1.90:3000
OPENROUTER_APP_NAME=Memory Engineering
OPENROUTER_VLM_MODEL=google/gemini-3.1-flash-lite-preview
OPENROUTER_LLM_MODEL=minimax/minimax-m2.5
AMAP_API_KEY=YOUR_AMAP_KEY

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
PORT=8000
BACKEND_RELOAD=false
FRONTEND_ORIGIN=http://10.60.1.90:3000
CORS_ALLOW_ORIGINS=

AUTH_SESSION_COOKIE_NAME=memory_session
AUTH_SESSION_DAYS=14
HIGH_SECURITY_MODE=false
ALLOW_SELF_REGISTRATION=true
COOKIE_SECURE=false

APP_ROLE=control-plane
WORKER_ORCHESTRATION_ENABLED=false

DATABASE_URL=mysql+pymysql://memory_app:YOUR_DB_PASSWORD@YOUR_RDS_ENDPOINT:3306/memory_engineering
SQL_ECHO=false

OBJECT_STORAGE_BUCKET=YOUR_S3_BUCKET
OBJECT_STORAGE_REGION=ap-southeast-1
OBJECT_STORAGE_PREFIX=tasks
OBJECT_STORAGE_ENDPOINT=
OBJECT_STORAGE_ACCESS_KEY_ID=
OBJECT_STORAGE_SECRET_ACCESS_KEY=
OBJECT_STORAGE_ADDRESSING_STYLE=auto

FACE_RECOGNITION_SRC_PATH=
FACE_MODEL_NAME=buffalo_l
FACE_MAX_SIDE=1920
FACE_DET_THRESHOLD=0.60
FACE_SIM_THRESHOLD=0.50
FACE_MIN_SIZE=48
FACE_MATCH_TOP_K=5
FACE_MATCH_MARGIN_THRESHOLD=0.03
FACE_MATCH_WEAK_DELTA=0.055
FACE_MATCH_MIN_QUALITY_GRAY_ZONE=0.40
FACE_MATCH_HIGH_QUALITY_THRESHOLD=0.40
FACE_PROVIDERS=CPUExecutionProvider
FACE_LANDMARKS_ENABLED=true

MEMORY_EXTERNAL_SINKS_ENABLED=true
MEMORY_REDIS_URL=
MEMORY_REDIS_PREFIX=memory

MEMORY_NEO4J_URI=neo4j://10.60.1.91:7687
MEMORY_NEO4J_USERNAME=neo4j
MEMORY_NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
MEMORY_NEO4J_DATABASE=neo4j

MEMORY_MILVUS_URI=http://10.60.1.91:19530
MEMORY_MILVUS_USER=
MEMORY_MILVUS_PASSWORD=
MEMORY_MILVUS_TOKEN=
MEMORY_MILVUS_DB_NAME=
MEMORY_MILVUS_COLLECTION=memory_segments
MEMORY_MILVUS_VECTOR_DIM=512
MEMORY_REAL_EMBEDDINGS_ENABLED=true
MEMORY_EMBEDDING_PROVIDER=fastembed
MEMORY_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
MEMORY_EMBEDDING_VERSION=v1
MEMORY_EMBEDDING_TIMEOUT_SECONDS=30
```

然后：

```bash
source .venv/bin/activate
python scripts/backfill_artifact_catalog.py

cd /opt/memory_engineering/frontend
npm run build

cd /opt/memory_engineering
```

### 5.1 直接前台验证

先用前台方式验证：

```bash
source .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

另一个终端：

```bash
cd /opt/memory_engineering/frontend
npm run start
```

### 5.2 连通性检查

从 app EC2 测 memory EC2：

```bash
nc -vz 10.60.1.91 7687
nc -vz 10.60.1.91 19530
```

从 app EC2 测 RDS：

```bash
nc -vz YOUR_RDS_ENDPOINT 3306
```

## 6. systemd（可选，验证通过后再做）

验证通过后，再用仓库现有 service 文件：

```bash
sudo cp /opt/memory_engineering/deploy/ec2/memory-engineering-backend.service /etc/systemd/system/
sudo cp /opt/memory_engineering/deploy/ec2/memory-engineering-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable memory-engineering-backend
sudo systemctl enable memory-engineering-frontend
sudo systemctl start memory-engineering-backend
sudo systemctl start memory-engineering-frontend
sudo systemctl status memory-engineering-backend --no-pager
sudo systemctl status memory-engineering-frontend --no-pager
```

## 7. 验证 checklist

### 7.1 app EC2

```bash
curl http://127.0.0.1:8000/api/health
```

应该看到：
- `app_version = v0317`
- `default_task_version = v0317`

### 7.2 前端

在办公网通过 VPN 访问：
- `http://10.60.1.90:3000`

### 7.3 Neo4j Browser

通过办公网访问：
- `http://10.60.1.91:7474`

### 7.4 Milvus

从 app EC2 看：

```bash
nc -vz 10.60.1.91 19530
curl http://10.60.1.91:9091/healthz
```

### 7.5 artifact catalog

任务跑完后确认：

```bash
curl http://127.0.0.1:8000/api/tasks/<task_id>/artifacts
```

## 8. 常见问题

### Neo4j 起不来
- 先看：
  - `docker logs memory-neo4j --tail 200`
- 常见原因：
  - 密码没配
  - 内存太小
  - 卷权限问题

### Milvus 起不来
- 先看：
  - `docker logs memory-milvus --tail 200`
  - `docker logs milvus-etcd --tail 200`
  - `docker logs milvus-minio --tail 200`

### app 连不上 memory
- 检查：
  - security groups
  - subnet route table
  - app `.env` 是否写的是私网 IP

### app 连不上 RDS
- 检查：
  - RDS SG 是否允许 `app-v0317-sg`
  - RDS 是否是 private
  - endpoint 和密码是否正确

## 9. 迁移建议

如果你后面要把旧环境迁进来：

1. 先在新环境跑通全新 `v0317`
2. 再决定是否迁老数据
3. 如果只迁 artifact 索引，可执行：

```bash
cd /opt/memory_engineering
source .venv/bin/activate
python scripts/backfill_artifact_catalog.py
```
