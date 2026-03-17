# AWS 部署方案（v0317+）

这套方案以 **结果准确性优先** 为前提，默认采用下面的拆分：

- `Frontend`: ECS Fargate
- `Control-plane Backend`: ECS Fargate
- `Worker`: EC2 Launch Template + 现有 `WorkerManager`
- `Artifact DB`: RDS MySQL，复用应用数据库并新增 `artifacts` 表
- `Artifact Object Store`: S3
- `Neo4j`: 单独 EC2 + Docker Compose
- `Milvus`: 单独 EC2 + Docker Compose（standalone）

## 推荐拓扑

```text
Browser
  ↓ HTTPS
ALB
  ├─ Frontend ECS Service
  └─ Control-plane ECS Service
          ├─ RDS MySQL (users/tasks/artifacts)
          ├─ S3 (task artifacts)
          ├─ Neo4j EC2 (Bolt 7687, private)
          ├─ Milvus EC2 (19530, private)
          └─ EC2 Workers (launch template)
```

## 为什么这么拆

- `RDS MySQL`
  - 当前代码已稳定支持 MySQL / PyMySQL
  - `artifacts` 表已经落地，适合作为 artifact catalog
- `S3`
  - 原图、缓存、result.json、debug 输出都属于对象资产，不适合放进数据库
- `Neo4j`
  - 负责 `facts + hypotheses`
- `Milvus`
  - 负责 `semantic evidence segments`
- `Worker EC2`
  - 继续复用当前 `WorkerManager` 与 launch template 逻辑，避免把 CPU/模型依赖塞进 Fargate

## 运行角色

### Frontend ECS
- 镜像：`deploy/aws/frontend.Dockerfile`
- 端口：`3000`
- 必填：
  - `NEXT_PUBLIC_API_BASE_URL`

### Control-plane ECS
- 镜像：`deploy/aws/backend.Dockerfile`
- 入口：`uvicorn backend.app:app`
- 端口：`8000`
- 关键环境：
  - `APP_ROLE=control-plane`
  - `DATABASE_URL=mysql+pymysql://...`
  - `OBJECT_STORAGE_BUCKET=...`
  - `OBJECT_STORAGE_REGION=ap-southeast-1`
  - `WORKER_ORCHESTRATION_ENABLED=true`
  - `AWS_REGION=...`
  - `MEMORY_NEO4J_URI=neo4j://...:7687`
  - `MEMORY_MILVUS_URI=http://...:19530`

### Worker EC2
- 入口：`uvicorn backend.worker_app:app`
- 角色：`APP_ROLE=worker`
- 本机保留 task 工作目录，结果同步回 control-plane

## Artifact DB 方案

### 表设计

应用数据库里新增 `artifacts` 表，记录：
- `artifact_id`
- `task_id`
- `user_id`
- `relative_path`
- `stage`
- `content_type`
- `size_bytes`
- `sha256`
- `storage_backend`
- `object_key`
- `asset_url`
- `metadata`

### 数据流

1. 上传图片/worker 生成产物
2. 写入 S3 或本地任务目录
3. 扫描任务目录生成 `asset_manifest`
4. `ArtifactCatalogStore.replace_task_artifacts()` 刷新 `artifacts` 表

### 为什么不只保留 `asset_manifest JSON`

- 便于前端 preview/debug
- 便于后续按任务/阶段/路径查询
- 便于 AWS 上对接 Athena/OpenSearch 前的第一层索引

## S3 配置

当前代码已支持两种模式：

### AWS 原生 S3（推荐）
- 只填：
  - `OBJECT_STORAGE_BUCKET`
  - `OBJECT_STORAGE_REGION`
- 不需要显式 access key
- 让 ECS task role / EC2 instance profile 直接读写 S3

### S3 兼容存储
- 继续使用：
  - `OBJECT_STORAGE_ENDPOINT`
  - `OBJECT_STORAGE_ACCESS_KEY_ID`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY`

## Neo4j AWS 配置

推荐单独一台 EC2，挂 `gp3` EBS：
- 实例起步：`m7i.xlarge` 或同级别
- EBS：`gp3 200GB+`
- 安全组：
  - `7687` 仅允许 control-plane / worker 子网访问
  - `7474` 仅允许堡垒机或办公网访问

部署方式：
- 使用 `deploy/aws/neo4j/docker-compose.yml`
- 环境模板：`deploy/aws/neo4j/neo4j.env.example`

应用侧配置：
- `MEMORY_NEO4J_URI=neo4j://<private-ip>:7687`
- `MEMORY_NEO4J_USERNAME=neo4j`
- `MEMORY_NEO4J_PASSWORD=...`
- `MEMORY_NEO4J_DATABASE=neo4j`

## Milvus AWS 配置

推荐单独一台 EC2 跑 standalone：
- 实例起步：`r7i.xlarge` 或同级别
- EBS：`gp3 300GB+`
- 安全组：
  - `19530` 仅允许 control-plane / worker 子网访问
  - `9091` 仅允许堡垒机或办公网访问
  - `9000/9001` 仅在运维排障时放开

部署方式：
- 使用 `deploy/aws/milvus/docker-compose.yml`
- 环境模板：`deploy/aws/milvus/milvus.env.example`

应用侧配置：
- `MEMORY_MILVUS_URI=http://<private-ip>:19530`
- `MEMORY_MILVUS_COLLECTION=memory_segments`
- `MEMORY_MILVUS_VECTOR_DIM=512`

## 安全组建议

### ALB SG
- `80/443` from internet
- egress 到 frontend/backend service

### ECS Service SG
- backend `8000` 只允许 ALB
- egress 到：
  - RDS `3306`
  - Neo4j `7687`
  - Milvus `19530`
  - S3 VPC endpoint 或公网 NAT

### Worker SG
- `9000` 只允许 control-plane SG
- egress 到：
  - Neo4j `7687`
  - Milvus `19530`
  - S3
  - OpenRouter/Gemini/Amap

## IAM 建议

### Control-plane task role
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`
- `s3:ListBucket`
- `secretsmanager:GetSecretValue` 或 `ssm:GetParameter`
- `ec2:RunInstances`
- `ec2:DescribeInstances`
- `ec2:TerminateInstances`
- `ec2:DescribeLaunchTemplates`
- `iam:PassRole`（如果 launch template 依赖 instance profile）

### Worker instance profile
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`
- `s3:ListBucket`
- `secretsmanager:GetSecretValue` 或 `ssm:GetParameter`

## 部署顺序

1. 创建 RDS MySQL
2. 创建 S3 bucket
3. 启动 Neo4j EC2
4. 启动 Milvus EC2
5. 构建并推送 frontend/backend 镜像到 ECR
6. 创建 ECS frontend/control-plane services
7. 配置 worker launch template
8. 运行：
   - `python scripts/backfill_artifact_catalog.py`
9. 用 `/api/tasks/{task_id}/artifacts` 和 `/api/tasks/{task_id}/memory/query` 验证

## 迁移检查项

- `DATABASE_URL` 已切到 RDS MySQL
- `OBJECT_STORAGE_BUCKET` 已切到 S3
- `artifacts` 表已创建
- `TaskAssetStore` 已用 IAM role 访问 S3
- `MEMORY_NEO4J_*` 已指向私网 Neo4j
- `MEMORY_MILVUS_*` 已指向私网 Milvus
- worker 的 `WORKER_SHARED_TOKEN` 与 control-plane 一致
