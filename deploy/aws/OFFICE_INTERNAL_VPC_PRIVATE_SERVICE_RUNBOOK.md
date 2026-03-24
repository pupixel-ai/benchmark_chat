# office-internal-vpc 新内网服务部署步骤

本文件按“给 AI agent 执行”的方式组织。执行前先收集输入，再按顺序执行，不要跳步，不要自行猜测服务名。

本文档用于指导在另一台电脑上，通过 AWS CLI 在现有 VPC `office-internal-vpc` 内部署一个新的私网服务，并在需要时为该服务新建一个私有 subnet。

本 runbook 默认区域固定为 `ap-southeast-1`（Singapore），下面所有 AWS CLI 示例都按这个区域书写。

适用前提：

- 服务运行在 AWS 内网 EC2（内网 VPS）上
- 服务不直接暴露公网 IP
- 运维入口优先使用 `AWS SSO + Systems Manager Session Manager`
- 服务如需访问外部 API，统一通过 NAT Gateway 或公司出口代理出网

## 0. AI 执行前必须先向用户确认的输入

执行前，先向用户确认下面这些值。其中 `SERVICE_NAME` 是必填，不允许 AI 自己命名。

- `SERVICE_NAME`
- `SERVICE_PORT`
- `SERVICE_AZ`
- `SERVICE_SUBNET_CIDR`
- 是否复用现有私网 route table
- 是否需要 NAT Gateway 出网
- 是否需要 internal ALB / 私有域名

如果用户还没有给服务命名，先问这一句，再继续执行：

```text
请先给这个 service 起一个名字，我会用它作为 AWS 资源命名前缀，例如 subnet、route table、security group 和实例名。
```

建议 AI 在真正执行 CLI 之前，把将要使用的输入整理成一段可回显的变量块，让用户做最后确认：

```bash
export AWS_PROFILE=office
export AWS_REGION=ap-southeast-1
export SERVICE_NAME=svc-new-internal
export SERVICE_PORT=8080
export SERVICE_AZ=ap-southeast-1a
export SERVICE_SUBNET_CIDR=10.60.32.0/24
export OFFICE_CIDR=192.168.1.0/24
```

## 1. 先明确是否真的需要新 subnet

不是每个新服务都需要新 subnet。优先复用现有私有 subnet，只有在以下情况之一成立时再新建：

- 现有私有 subnet 可用 IP 不足
- 需要把新服务与现有服务做网络隔离
- 需要单独的路由策略、NACL 或 VPC endpoint 策略
- 需要把服务放到新的可用区
- 计划后续挂内部 ALB/NLB、Auto Scaling、多个 ENI，现有 subnet 太小

如果现有私有 subnet 已满足容量、路由和隔离要求，直接复用会更简单。

## 2. 在另一台电脑上准备登录能力

推荐使用 AWS SSO，而不是长期 `access key`。

### 2.1 安装工具

- 安装 AWS CLI v2
- 安装 Session Manager plugin
- 安装 `jq`

### 2.2 配置 AWS SSO

```bash
aws configure sso --profile office
aws sso login --profile office
aws sts get-caller-identity --profile office
```

你至少需要这些权限：

- `ec2:Describe*`
- `ec2:CreateSubnet`
- `ec2:CreateRouteTable`
- `ec2:AssociateRouteTable`
- `ec2:CreateSecurityGroup`
- `ec2:AuthorizeSecurityGroupIngress`
- `ec2:AuthorizeSecurityGroupEgress`
- `ec2:RunInstances`
- `ec2:CreateTags`
- `iam:PassRole`
- `ssm:StartSession`

如果你们团队不用 SSO，也可以用临时凭证：

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `AWS_REGION`

只给 `access key id` 不够，至少还要配套 `secret access key`，很多场景还必须带 `session token`。

## 3. 先盘点 `office-internal-vpc`

先确定 VPC、已有 subnet、路由表、NAT、VPC endpoints，避免新建出和现网冲突的网段。

```bash
export AWS_PROFILE=office
export AWS_REGION=ap-southeast-1

VPC_ID=$(aws ec2 describe-vpcs \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --filters "Name=tag:Name,Values=office-internal-vpc" \
  --query 'Vpcs[0].VpcId' \
  --output text)

aws ec2 describe-vpcs \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --vpc-ids "$VPC_ID" \
  --query 'Vpcs[0].{VpcId:VpcId,CIDRs:CidrBlockAssociationSet[*].CidrBlock,DnsHostnames:EnableDnsHostnames,DnsSupport:EnableDnsSupport}' \
  --output table

aws ec2 describe-subnets \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[].{Name:Tags[?Key==`Name`]|[0].Value,SubnetId:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock,AvailableIPs:AvailableIpAddressCount,MapPublicIp:MapPublicIpOnLaunch}' \
  --output table

aws ec2 describe-route-tables \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'RouteTables[].{RouteTableId:RouteTableId,Main:Associations[0].Main,Routes:Routes[*].[DestinationCidrBlock,GatewayId,NatGatewayId,TransitGatewayId]}' \
  --output json

aws ec2 describe-vpc-endpoints \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'VpcEndpoints[].{Id:VpcEndpointId,Service:ServiceName,Type:VpcEndpointType,State:State}' \
  --output table
```

盘点完后先记录这几项：

- VPC 主 CIDR
- 已占用的 subnet CIDR
- 现有私网 route table
- 是否已有 NAT Gateway
- 是否已有 SSM / S3 / ECR / Secrets Manager / CloudWatch Logs 等 VPC endpoint

## 4. 规划新的 subnet CIDR

新 subnet 的 CIDR 必须同时满足：

- 在 `office-internal-vpc` 的 VPC CIDR 范围内
- 不与现有任意 subnet 重叠
- 预留足够 IP 给实例、ENI、endpoint、扩容

常用经验：

- 单机小服务：`/27` 或 `/26`
- 预期会扩容或挂多个 endpoint：`/24`
- 尽量不要把 subnet 规划得过小，否则后续很容易被 ENI 和 endpoint 吃掉

如果 VPC 主网段是 `10.60.0.0/16`，那么新 subnet 可以是类似 `10.60.32.0/24` 这样的地址，但这只是示例，必须以实际盘点结果为准。

建议同时确定：

- 服务名：例如 `svc-memory-worker`
- 端口：例如 `8080`
- 可用区：例如 `ap-southeast-1a`
- 访问方：哪些 SG 或子网可以访问它
- 出网方式：`NAT Gateway`、`公司代理` 或 `完全不出网`

## 5. 创建新的私有 subnet

```bash
export SERVICE_NAME=svc-new-internal
export SERVICE_PORT=8080
export SERVICE_AZ=ap-southeast-1a
export SERVICE_SUBNET_CIDR=10.60.32.0/24
export OFFICE_CIDR=192.168.1.0/24

SUBNET_ID=$(aws ec2 create-subnet \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --vpc-id "$VPC_ID" \
  --availability-zone "$SERVICE_AZ" \
  --cidr-block "$SERVICE_SUBNET_CIDR" \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${SERVICE_NAME}-private-subnet},{Key=Role,Value=private-service}]" \
  --query 'Subnet.SubnetId' \
  --output text)

aws ec2 modify-subnet-attribute \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --subnet-id "$SUBNET_ID" \
  --no-map-public-ip-on-launch
```

这一步完成后，subnet 只是创建出来了，还没有路由策略。

## 6. 配置路由

### 方案 A：复用现有私网 route table

如果现有私网 route table 已经满足要求，例如：

- 只保留 `local` 路由
- 或者已有 `0.0.0.0/0 -> NAT Gateway`
- 或者已经挂好了所需 VPC endpoint

那么可以直接把新 subnet 关联进去：

```bash
export PRIVATE_RT_ID=rtb-xxxxxxxx

aws ec2 associate-route-table \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --subnet-id "$SUBNET_ID" \
  --route-table-id "$PRIVATE_RT_ID"
```

### 方案 B：给新服务单独建 route table

如果你希望这个服务独立控制出网、VPC endpoint 或未来 ACL，就新建 route table：

```bash
SERVICE_RT_ID=$(aws ec2 create-route-table \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --vpc-id "$VPC_ID" \
  --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=${SERVICE_NAME}-private-rt}]" \
  --query 'RouteTable.RouteTableId' \
  --output text)

aws ec2 associate-route-table \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --subnet-id "$SUBNET_ID" \
  --route-table-id "$SERVICE_RT_ID"
```

如果服务需要访问公网 API 或拉外部依赖，再加默认路由到 NAT：

```bash
export NAT_GW_ID=nat-xxxxxxxx

aws ec2 create-route \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --route-table-id "$SERVICE_RT_ID" \
  --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id "$NAT_GW_ID"
```

注意：

- 私有 subnet 不要直接绑定公网 IP
- 不要把 `0.0.0.0/0` 指到 Internet Gateway，除非你明确要把它变成 public subnet

## 7. 如果不想走 NAT，补齐 VPC endpoints

如果实例没有公网 IP，且也不走 NAT，那么要想使用 SSM、拉私有镜像、读 S3、取 secrets，就需要相应的 VPC endpoint。

常见最小集合：

- Systems Manager：
  - `com.amazonaws.${AWS_REGION}.ssm`
  - `com.amazonaws.${AWS_REGION}.ssmmessages`
  - `com.amazonaws.${AWS_REGION}.ec2messages`
- S3：Gateway endpoint
- ECR：
  - `com.amazonaws.${AWS_REGION}.ecr.api`
  - `com.amazonaws.${AWS_REGION}.ecr.dkr`
- Secrets Manager：`com.amazonaws.${AWS_REGION}.secretsmanager`
- CloudWatch Logs：`com.amazonaws.${AWS_REGION}.logs`
- KMS：`com.amazonaws.${AWS_REGION}.kms`

如果你的服务会：

- 从 ECR 拉镜像
- 从 S3 拉部署包
- 用 SSM 远程运维
- 从 Secrets Manager 或 Parameter Store 读配置

那么这些 endpoint 通常都值得提前建好。

## 8. 创建安全组

新服务建议单独一个 SG，不要复用过大的通用 SG。

```bash
SG_ID=$(aws ec2 create-security-group \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-name "${SERVICE_NAME}-sg" \
  --description "Security group for ${SERVICE_NAME}" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' \
  --output text)
```

只放开明确来源。例子：

- 服务端口只允许内网调用方的安全组访问
- 办公网段 `192.168.1.0/24` 允许访问服务端口，方便办公室直接访问
- 不允许 `0.0.0.0/0` 入站
- 如果用 SSM，就不需要开放 `22`

例如只允许某个调用方 SG 访问 `8080`：

```bash
export CALLER_SG_ID=sg-xxxxxxxx

aws ec2 authorize-security-group-ingress \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-id "$SG_ID" \
  --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":${SERVICE_PORT},\"ToPort\":${SERVICE_PORT},\"UserIdGroupPairs\":[{\"GroupId\":\"${CALLER_SG_ID}\"}]}]"
```

再加入办公室网段的访问规则：

```bash
export OFFICE_CIDR=192.168.1.0/24

aws ec2 authorize-security-group-ingress \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-id "$SG_ID" \
  --ip-permissions "[{\"IpProtocol\":\"tcp\",\"FromPort\":${SERVICE_PORT},\"ToPort\":${SERVICE_PORT},\"IpRanges\":[{\"CidrIp\":\"${OFFICE_CIDR}\",\"Description\":\"Office network access\"}]}]"
```

如果这个服务只允许办公室访问，而不允许其他内部调用方，就不要再添加额外的 `CALLER_SG_ID` 入站规则。

如果需要出网到 NAT 或 endpoint，通常保留默认 egress 即可；如果你们要求严格最小权限，再按目的网段或端口收紧。

## 9. 创建实例角色

内网 VPS 不建议塞长期 AK/SK。推荐给 EC2 挂 instance profile。

最少角色能力通常包括：

- `AmazonSSMManagedInstanceCore`
- 读取配置或密钥：
  - `secretsmanager:GetSecretValue`
  - 或 `ssm:GetParameter`
- 访问对象存储：
  - `s3:GetObject`
  - `s3:PutObject`
  - `s3:ListBucket`
- 如果要拉 ECR 镜像，还需要 ECR 读取权限

如果这台机只负责跑应用，不要给它管理整套 EC2 的权限。

## 10. 启动新的内网 VPS

建议配置：

- 子网：刚创建的 `SUBNET_ID`
- 不分配公网 IP
- 挂上新建的 `SG_ID`
- 挂上 instance profile
- 根盘使用 `gp3`
- 开启 IMDSv2

启动前确认：

- AMI 是否已内置 SSM Agent
- 是否需要 Docker / Docker Compose / systemd
- 是否要从 ECR 拉镜像，还是直接拉 Git 仓库代码

一个常见部署方式是：

- EC2 作为长期内网服务宿主机
- 用 `docker compose` 起服务
- 用 `systemd` 管理 compose 或二进制进程

## 11. 首次接入这台内网 VPS

如果已经配置了 SSM：

```bash
aws ssm start-session \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --target i-xxxxxxxx
```

如果公司仍使用堡垒机：

```bash
ssh -J ec2-user@<bastion-public-ip> ubuntu@<private-ip>
```

推荐优先 SSM，因为：

- 不需要开放 `22`
- 不需要管理 SSH key 分发
- 审计更容易

## 12. 发布服务

你需要先确定服务是如何交付的：

- `Git pull + venv/systemd`
- `Docker image + docker compose`
- `ECR image + systemd/pull`

以 Docker Compose 为例：

```bash
sudo mkdir -p /opt/${SERVICE_NAME}
cd /opt/${SERVICE_NAME}

# 放置 compose.yaml 和 .env
docker compose pull
docker compose up -d
docker compose ps
```

如果服务依赖外部 API，例如 LLM、地图、支付、Webhook：

- 私网实例仍然需要出网能力
- 优先走 NAT Gateway 或公司 egress proxy
- 不建议为了“能访问外网”而直接给服务机绑定公网 IP

## 13. 如果要给别的内网服务调用

有三种常见暴露方式：

### 方式 A：直接用私网 IP

适合：

- 调用方很少
- 没有高可用要求
- 你能接受实例更换后 IP 变化

### 方式 B：Route 53 Private Hosted Zone

给服务一个内网域名，例如：

- `svc-new-internal.office.local`

适合：

- 你希望调用方不关心实例 IP
- 后续可能换机或做蓝绿发布

### 方式 C：Internal ALB 或 NLB

适合：

- 需要多实例
- 需要健康检查
- 需要稳定入口
- 可能后续做灰度和自动扩容

如果你一开始就知道服务会被多个系统调用，建议直接考虑 internal ALB，而不是把单台机器 IP 暴露给所有调用方。

## 14. 验证清单

部署完成后至少验证这些：

- `describe-subnets` 中新 subnet 已存在且无公网 IP 自动分配
- route table 关联正确
- 实例已启动在目标 subnet 中
- SSM 可以正常连上实例
- 服务端口只对预期来源开放
- 从调用方机器或调用方子网可以访问服务
- 如果需要出网，实例可以访问外部依赖
- 日志能进入 CloudWatch 或本地日志目录
- 如果用了 Secrets Manager / S3 / ECR / SSM，相关 endpoint 或 NAT 实际可用

## 15. 回滚步骤

如果新服务上线失败，按这个顺序回滚：

1. 停止服务进程或 `docker compose down`
2. 从调用方配置里摘除这个服务
3. 如有 internal ALB target group，先摘流量
4. 终止 EC2 实例
5. 删除不再使用的 SG
6. 解除 route table 关联
7. 删除新建 route table
8. 删除新建 subnet

不要先删 subnet，再处理实例和网卡，否则通常会因为 ENI 仍占用而删除失败。

## 16. 最终建议

对 `office-internal-vpc` 里的新私网服务，推荐默认方案是：

1. 另一台电脑使用 `AWS SSO` 登录
2. 先向用户确认 `SERVICE_NAME`，不要自行命名
3. 先盘点现有私有 subnet 是否可复用
4. 只有在容量、隔离或路由需要时才新建 subnet
5. 新 subnet 保持 private，不分配公网 IP
6. 安全组里显式加入办公室网段 `192.168.1.0/24` 到服务端口
7. 运维入口优先走 `SSM`
8. 服务实例使用 IAM role，不放长期 AK/SK
9. 外部依赖统一走 NAT 或公司出口代理
10. 如果服务未来会被多个系统依赖，尽早上 internal ALB + 私有域名

## 参考

- AWS VPC 子网说明：[Subnets for your VPC](https://docs.aws.amazon.com/vpc/latest/userguide/configure-subnets.html)
- AWS 创建子网：[Create a subnet](https://docs.aws.amazon.com/vpc/latest/userguide/create-subnets.html)
- AWS 路由表与子网关联：[Subnet route tables](https://docs.aws.amazon.com/vpc/latest/userguide/subnet-route-tables.html)
- AWS 创建路由表：[Create a route table for your VPC](https://docs.aws.amazon.com/vpc/latest/userguide/create-vpc-route-table.html)
- AWS Systems Manager 私网访问与 VPC endpoints：[Improve the security of EC2 instances by using VPC endpoints for Systems Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-create-vpc.html)
- AWS Session Manager 实例角色：[Step 2: Verify or add instance permissions for Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-instance-profile.html)
- AWS EC2 IAM role 说明：[IAM roles for Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)
- AWS ECR 私网 endpoint：[Amazon ECR interface VPC endpoints](https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html)
