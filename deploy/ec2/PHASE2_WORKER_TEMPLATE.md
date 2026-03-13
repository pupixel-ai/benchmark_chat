# Phase 2 Worker Launch Template

本文说明如何给 Phase 2 的 `worker` 准备 AWS 启动模板，目标是：

- `control-plane` 对外提供登录、任务列表、删除按钮
- 每个任务启动一台私有 `worker`
- 敏感原图和中间产物只落在 `instance store`
- 删除任务时，终止对应 worker

当前仓库里的 Phase 2 worker 相关模板文件：

- `deploy/ec2/worker.env.example`
- `deploy/ec2/memory-engineering-worker.service`
- `deploy/ec2/worker-user-data.sh.example`

## 1. 推荐前提

推荐先做一台 **worker golden AMI**，不要让每台 worker 首次启动时才去安装依赖。

原因：

- `insightface` / `onnxruntime` 初始化可能触发模型准备
- 临时 worker 若在冷启动时安装依赖，会拖慢上传后的首个任务
- 你已经要把敏感图片放到 `instance store`，更适合将系统和依赖提前烘焙进 AMI

推荐 worker AMI 至少包含：

- Ubuntu 24.04
- `/opt/memory_engineering`
- `.venv`
- `pip install -r requirements.txt`
- `deploy/ec2/memory-engineering-worker.service`

## 2. Worker 实例类型

worker 必须选 **带 instance store 的机型**，不要选只有 EBS 的通用机型。

推荐起步：

- `i4i.large`
- `i4i.xlarge`

选择原则：

- root volume 继续用加密 EBS
- 敏感任务数据只写到 `instance store`
- `WORKER_TASK_ROOT` 固定为 `/mnt/secure-tasks`

## 3. IAM Role

### 3.1 Worker Instance Role

worker 的 IAM role 建议最小化：

- 附加 AWS 托管策略：`AmazonSSMManagedInstanceCore`

如果你要把日志发到 CloudWatch，再额外加：

- `CloudWatchAgentServerPolicy`

worker 本身 **不需要** 默认拿到：

- `AmazonS3FullAccess`
- 广泛的 `EC2:*`
- 广泛的 `IAM:*`

### 3.2 Control-Plane Role

因为当前 Phase 2 scaffolding 里的 `backend/worker_manager.py` 会从 control-plane 调 `RunInstances` / `DescribeInstances` / `TerminateInstances`，所以 control-plane 还需要一个单独角色，至少允许：

- `ec2:RunInstances`
- `ec2:DescribeInstances`
- `ec2:TerminateInstances`
- `ec2:CreateTags`
- `ec2:DescribeSubnets`
- `ec2:DescribeSecurityGroups`
- `ec2:DescribeLaunchTemplates`
- `iam:PassRole`（只限 worker instance profile）

## 4. Security Groups

建议至少拆成三组：

### 4.1 `memory-control-plane-sg`

用途：

- 挂在 control-plane EC2

至少需要：

- 出站允许访问 worker 的 `9000`

### 4.2 `memory-worker-sg`

用途：

- 挂在每台 worker

入站：

- `TCP 9000`，来源仅 `memory-control-plane-sg`

不要开放：

- `22`
- `80`
- `443`
- `9000` 来自公网

出站：

- 如果你走 NAT：允许 `443` 到外网
- 如果你只走 VPC endpoint：允许 `443` 到对应 endpoint SG

### 4.3 `memory-endpoints-sg`

用途：

- 挂在 Interface VPC Endpoints（SSM 等）

入站：

- `TCP 443`，来源 `memory-worker-sg`
- `TCP 443`，来源 `memory-control-plane-sg`

## 5. Private Subnet

建议：

- worker 放单独 private subnets
- `Auto-assign public IPv4` 关闭
- route table 不直连 IGW

两种网络模式选一种：

### 模式 A：最省事

- private subnet -> NAT Gateway
- worker 通过 NAT 出网
- 同时创建 SSM endpoint

### 模式 B：更收敛

- private subnet 不配公网出站
- 创建这些 Interface VPC Endpoints：
  - `ssm`
  - `ssmmessages`
  - `ec2messages`

如果你还要：

- CloudWatch Logs：再加 `logs`
- KMS：再加 `kms`

如果 worker AMI 已经预装好仓库、依赖和模型，模式 B 会更贴近“最小外连面”。

## 6. Launch Template

在 EC2 控制台创建 Launch Template，关键项这样填：

- `Template name`: `memory-worker-template`
- `AMI`: 你的 worker golden AMI
- `Instance type`: `i4i.large` 或更高
- `Key pair`: 不选
- `Subnet`: 不在模板里写死也可以，由 control-plane 指定
- `Auto-assign public IP`: `Disable`
- `Security groups`: `memory-worker-sg`
- `IAM instance profile`: worker role
- `Shutdown behavior`: `Terminate`
- `Detailed monitoring`: 可选
- `Metadata version`: `IMDSv2 required`
- `User data`: 使用 `deploy/ec2/worker-user-data.sh.example`

### User Data 占位符

把下面三个值替换掉：

- `__AWS_REGION__`
- `__WORKER_SHARED_TOKEN__`
- `__CONTROL_PLANE_ORIGIN__`

`__CONTROL_PLANE_ORIGIN__` 暂时可以先填：

- `http://YOUR_CONTROL_PLANE_PRIVATE_HOST`

等你审批通过 HTTPS 后，再改成正式域名。

## 7. Worker systemd

worker 启动命令已经写在：

- `deploy/ec2/memory-engineering-worker.service`

核心启动命令是：

```bash
/opt/memory_engineering/.venv/bin/uvicorn backend.worker_app:app --host 0.0.0.0 --port 9000
```

实际 systemd 会从 `.worker.env` 读 `WORKER_INTERNAL_PORT`。

## 8. Worker AMI 制作建议

推荐流程：

1. 启动一台临时 Ubuntu 24.04 builder
2. 安装仓库和依赖
3. 复制 `deploy/ec2/worker.env.example` 为 `.worker.env`
4. 安装 `memory-engineering-worker.service`
5. 手动跑一次：

```bash
cd /opt/memory_engineering
source .venv/bin/activate
WORKER_INTERNAL_PORT=9000 WORKER_SHARED_TOKEN=test-token APP_ROLE=worker python -m uvicorn backend.worker_app:app --host 127.0.0.1 --port 9000
```

6. 验证：

```bash
curl -H 'Authorization: Bearer test-token' http://127.0.0.1:9000/internal/health
```

7. 停机并创建 AMI

## 9. Control-Plane 环境变量

control-plane 这一侧至少要补：

```env
WORKER_ORCHESTRATION_ENABLED=true
AWS_REGION=your-region
WORKER_LAUNCH_TEMPLATE_ID=lt-xxxxxxxx
WORKER_LAUNCH_TEMPLATE_VERSION=$Default
WORKER_INTERNAL_PORT=9000
WORKER_SHARED_TOKEN=replace_with_the_same_secret
WORKER_INSTANCE_NAME_PREFIX=memory-worker
RESULT_TTL_HOURS=24
WORKER_POLL_SECONDS=3
WORKER_BOOT_TIMEOUT_SECONDS=300
```

如果你暂时不想在 launch template 里固定 subnet，也可以继续让 control-plane 通过：

```env
WORKER_SUBNET_IDS=subnet-aaa,subnet-bbb
```

来决定落在哪个 private subnet。

## 10. 删除语义

当前第一批 backend scaffolding 已经把删除路径改成：

- control-plane 标记 `delete_state=requested`
- 尝试通知 worker 删除任务目录
- control-plane 调 `TerminateInstances`

最终真正的“不可恢复删除”依赖于：

- worker 使用的是 `instance store`
- 删除时实例被终止
- 不要对 worker 做快照、AMI、Recycle Bin 保留或其他备份

## 11. 当前阶段最推荐的执行顺序

1. 先做 worker role
2. 再做 worker SG 和 private subnet
3. 准备 worker golden AMI
4. 创建 worker launch template
5. 在 control-plane `.env` 里打开 `WORKER_ORCHESTRATION_ENABLED=true`
6. 用一组测试图片验证：
   - 新建任务
   - worker 被拉起
   - 图片可显示
   - 点击删除后 worker 被终止
